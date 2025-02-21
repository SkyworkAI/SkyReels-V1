import logging
import os
import threading
import time
from datetime import timedelta
from typing import Any
from typing import Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import HunyuanVideoTransformer3DModel
from PIL import Image
from torchao.quantization import float8_weight_only
from torchao.quantization import quantize_
from transformers import LlamaModel

from . import TaskType
from .offload import Offload
from .offload import OffloadConfig
from .pipelines import SkyreelsVideoPipeline

logger = logging.getLogger("SkyreelsVideoInfer")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    f"%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s] - %(message)s"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class SkyReelsVideoSingleGpuInfer:
    def _load_model(
        self,
        model_id: str,
        base_model_id: str = "hunyuanvideo-community/HunyuanVideo",
        quant_model: bool = True,
        gpu_device: str = "cuda:0",
    ) -> SkyreelsVideoPipeline:
        logger.info(f"load model model_id:{model_id} quan_model:{quant_model} gpu_device:{gpu_device}")
        text_encoder = LlamaModel.from_pretrained(
            base_model_id,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        ).to("cpu")
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id,
            # subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device="cpu",
        ).to("cpu")
        if quant_model:
            quantize_(text_encoder, float8_weight_only(), device=gpu_device)
            text_encoder.to("cpu")
            torch.cuda.empty_cache()
            quantize_(transformer, float8_weight_only(), device=gpu_device)
            transformer.to("cpu")
            torch.cuda.empty_cache()
        pipe = SkyreelsVideoPipeline.from_pretrained(
            base_model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch.bfloat16,
        ).to("cpu")
        pipe.vae.enable_tiling()
        torch.cuda.empty_cache()
        return pipe

    def __init__(
        self,
        task_type: TaskType,
        model_id: str,
        quant_model: bool = True,
        local_rank: int = 0,
        world_size: int = 1,
        is_offload: bool = True,
        offload_config: OffloadConfig = OffloadConfig(),
        enable_cfg_parallel: bool = True,
    ):
        self.task_type = task_type
        self.gpu_rank = local_rank

        nccl_available = dist.is_nccl_available()
        communication_backend = "nccl" if nccl_available else "gloo"

        dist.init_process_group(
            backend=communication_backend,
            init_method="tcp://127.0.0.1:23456",
            timeout=timedelta(seconds=600),
            world_size=world_size,
            rank=local_rank,
        )
        os.environ["LOCAL_RANK"] = str(local_rank)
        logger.info(f"rank:{local_rank} Distributed backend: {dist.get_backend()}")
        torch.cuda.set_device(dist.get_rank())
        torch.backends.cuda.enable_cudnn_sdp(False)
        gpu_device = f"cuda:{dist.get_rank()}"

        self.pipe: SkyreelsVideoPipeline = self._load_model(
            model_id=model_id, quant_model=quant_model, gpu_device=gpu_device
        )

        if nccl_available:
            from para_attn.context_parallel import init_context_parallel_mesh
            from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
            from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

            max_batch_dim_size = 2 if enable_cfg_parallel and world_size > 1 else 1
            max_ulysses_dim_size = int(world_size / max_batch_dim_size)
            logger.info(f"max_batch_dim_size: {max_batch_dim_size}, max_ulysses_dim_size:{max_ulysses_dim_size}")

            mesh = init_context_parallel_mesh(
                self.pipe.device.type,
                max_ring_dim_size=1,
                max_batch_dim_size=max_batch_dim_size,
            )
            parallelize_pipe(self.pipe, mesh=mesh)
            parallelize_vae(self.pipe.vae, mesh=mesh._flatten())
        else:
            assert world_size == 1, "Multi GPU distribution only supported on platforms supporting nccl"

        if is_offload:
            Offload.offload(
                pipeline=self.pipe,
                config=offload_config,
            )
        else:
            self.pipe.to(gpu_device)

        if offload_config.compiler_transformer:
            torch._dynamo.config.suppress_errors = True
            os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{offload_config.compiler_cache}_{world_size}"
            self.pipe.transformer = torch.compile(
                self.pipe.transformer,
                mode="max-autotune-no-cudagraphs",
                dynamic=True,
            )
            self.warm_up()

    def warm_up(self):
        init_kwargs = {
            "prompt": "A woman is dancing in a room",
            "height": 544,
            "width": 960,
            "guidance_scale": 6,
            "num_inference_steps": 1,
            "negative_prompt": "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
            "num_frames": 97,
            "generator": torch.Generator("cuda").manual_seed(42),
            "embedded_guidance_scale": 1.0,
        }
        if self.task_type == TaskType.I2V:
            init_kwargs["image"] = Image.new("RGB", (544, 960), color="black")
        self.pipe(**init_kwargs)

    def damon_inference(self, request_queue: mp.Queue, response_queue: mp.Queue):
        response_queue.put(f"rank:{self.gpu_rank} ready")
        logger.info(f"rank:{self.gpu_rank} finish init pipe")
        while True:
            logger.info(f"rank:{self.gpu_rank} waiting for request")
            kwargs = request_queue.get()
            logger.info(f"rank:{self.gpu_rank} kwargs: {kwargs}")
            if "seed" in kwargs:
                kwargs["generator"] = torch.Generator("cuda").manual_seed(kwargs["seed"])
                del kwargs["seed"]
            start_time = time.time()
            assert (self.task_type == TaskType.I2V and "image" in kwargs) or self.task_type == TaskType.T2V
            out = self.pipe(**kwargs).frames[0]
            logger.info(f"rank:{dist.get_rank()} inference time: {time.time() - start_time}")
            if dist.get_rank() == 0:
                response_queue.put(out)


def single_gpu_run(
    rank,
    task_type: TaskType,
    model_id: str,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    quant_model: bool = True,
    world_size: int = 1,
    is_offload: bool = True,
    offload_config: OffloadConfig = OffloadConfig(),
    enable_cfg_parallel: bool = True,
):
    pipe = SkyReelsVideoSingleGpuInfer(
        task_type=task_type,
        model_id=model_id,
        quant_model=quant_model,
        local_rank=rank,
        world_size=world_size,
        is_offload=is_offload,
        offload_config=offload_config,
        enable_cfg_parallel=enable_cfg_parallel,
    )
    pipe.damon_inference(request_queue, response_queue)


class SkyReelsVideoInfer:
    def __init__(
        self,
        task_type: TaskType,
        model_id: str,
        quant_model: bool = True,
        world_size: int = 1,
        is_offload: bool = True,
        offload_config: OffloadConfig = OffloadConfig(),
        enable_cfg_parallel: bool = True,
    ):
        self.world_size = world_size
        smp = mp.get_context("spawn")
        self.REQ_QUEUES: mp.Queue = smp.Queue()
        self.RESP_QUEUE: mp.Queue = smp.Queue()
        assert self.world_size > 0, "gpu_num must be greater than 0"
        spawn_thread = threading.Thread(
            target=self.lauch_single_gpu_infer,
            args=(task_type, model_id, quant_model, world_size, is_offload, offload_config, enable_cfg_parallel),
            daemon=True,
        )
        spawn_thread.start()
        logger.info(f"Started multi-GPU thread with GPU_NUM: {world_size}")
        print(f"Started multi-GPU thread with GPU_NUM: {world_size}")
        # Block and wait for the prediction process to start
        for _ in range(world_size):
            msg = self.RESP_QUEUE.get()
            logger.info(f"launch_multi_gpu get init msg: {msg}")
            print(f"launch_multi_gpu get init msg: {msg}")

    def lauch_single_gpu_infer(
        self,
        task_type: TaskType,
        model_id: str,
        quant_model: bool = True,
        world_size: int = 1,
        is_offload: bool = True,
        offload_config: OffloadConfig = OffloadConfig(),
        enable_cfg_parallel: bool = True,
    ):
        mp.spawn(
            single_gpu_run,
            nprocs=world_size,
            join=True,
            daemon=True,
            args=(
                task_type,
                model_id,
                self.REQ_QUEUES,
                self.RESP_QUEUE,
                quant_model,
                world_size,
                is_offload,
                offload_config,
                enable_cfg_parallel,
            ),
        )
        logger.info(f"finish lanch multi gpu infer, world_size:{world_size}")

    def inference(self, kwargs: Dict[str, Any]):
        # put request to singlegpuinfer
        for _ in range(self.world_size):
            self.REQ_QUEUES.put(kwargs)
        return self.RESP_QUEUE.get()
