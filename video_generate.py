import argparse
import os
import time
import random

from diffusers.utils import export_to_video
from diffusers.utils import load_image

from skyreelsinfer import TaskType
from skyreelsinfer.offload import OffloadConfig
from skyreelsinfer.skyreels_video_infer import SkyReelsVideoInfer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V1-Hunyuan-T2V")
    parser.add_argument("--outdir", type=str, default="skyreels")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=-1) # 20250224 42 default changed to -1 otherwise negative seed never triggers
    parser.add_argument("--prompt", type=str, default="FPS-24, A 3D model of a 1800s victorian house.")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
    )
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--video_num", type=int, default=1) # 20250224 pftq: default 1 video instead of 2
    parser.add_argument("--task_type", type=str, default="t2v", choices=["t2v", "i2v"])
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--embedded_guidance_scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quant", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--high_cpu_memory", action="store_true")
    parser.add_argument("--parameters_level", action="store_true")
    parser.add_argument("--compiler_transformer", action="store_true")
    parser.add_argument("--sequence_batch", action="store_true")
    parser.add_argument("--mbps", type=float, default=7)

    args = parser.parse_args()

    out_dir = f"results/{args.outdir}"
    os.makedirs(out_dir, exist_ok=True)

    if args.task_type == "i2v":
        image = load_image(args.image)
    
    predictor = SkyReelsVideoInfer(
        task_type=TaskType.I2V if args.task_type == "i2v" else TaskType.T2V,
        model_id=args.model_id,
        quant_model=args.quant,
        world_size=args.gpu_num,
        is_offload=args.offload,
        offload_config=OffloadConfig(
            high_cpu_memory=args.high_cpu_memory,
            parameters_level=args.parameters_level,
            compiler_transformer=args.compiler_transformer,
        ),
        enable_cfg_parallel=args.guidance_scale > 1.0 and not args.sequence_batch,
    )
    print("finish pipeline init")

    for idx in range(args.video_num): # 20250224 pftq: for loop needs to start earlier before seed is set
        
        if args.seed == -1 or idx > 0: # 20250224 pftq: seed argument ignored if asking for more than one video
            random.seed(time.time())
            args.seed = int(random.randrange(4294967294))
        
        kwargs = {
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
            "guidance_scale": args.guidance_scale,
            "embedded_guidance_scale": args.embedded_guidance_scale,
            "negative_prompt": args.negative_prompt,
            "cfg_for": args.sequence_batch,
        }
        if args.task_type == "i2v":
            kwargs["image"] = image
    
    
        #20250223 pftq: customizable bitrate
        # Function to check if FFmpeg is installed
        import subprocess  # For FFmpeg functionality
        import numpy as np  # For frame conversion
        import cv2  # For OpenCV fallback
        def is_ffmpeg_installed():
            try:
                subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
        
        # FFmpeg-based video saving with bitrate control
        def save_video_with_ffmpeg(frames, output_path, fps, bitrate_mbps):
            frames = [np.array(frame) for frame in frames]
            height, width, _ = frames[0].shape
            bitrate = f"{bitrate_mbps}M"
            cmd = [
                "ffmpeg",
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "rgb24",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264",
                "-b:v", bitrate,
                "-pix_fmt", "yuv420p",
                "-preset", "medium",
                output_path
            ]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            for frame in frames:
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            process.wait()
            stderr_output = process.stderr.read().decode()
            if process.returncode != 0:
                print(f"FFmpeg error: {stderr_output}")
            else:
                print(f"Video saved to {output_path} with FFmpeg")
        
        # Fallback OpenCV-based video saving
        def save_video_with_opencv(frames, output_path, fps, bitrate_mbps):
            frames = [np.array(frame) for frame in frames]
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            # Note: cv2.CAP_PROP_BITRATE is not supported, so bitrate_mbps is ignored
            for frame in frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
                writer.write(frame)
            writer.release()
            print(f"Video saved to {output_path} with OpenCV (bitrate control unavailable)")
        
        # Wrapper to choose between FFmpeg and OpenCV
        def save_video_with_quality(frames, output_path, fps, bitrate_mbps):
            if is_ffmpeg_installed():
                save_video_with_ffmpeg(frames, output_path, fps, bitrate_mbps)
            else:
                print("FFmpeg not found. Falling back to OpenCV (bitrate not customizable).")
                save_video_with_opencv(frames, output_path, fps, bitrate_mbps)
    
        
        
        output = predictor.inference(kwargs)
        #video_out_file = f"{args.prompt[:100].replace('/','')}_{args.seed}_{idx}.mp4"
        #export_to_video(output, f"{out_dir}/{video_out_file}", fps=args.fps)
    
        #20250223 pftq: More useful filename and higher customizable bitrate
        from datetime import datetime
        now = datetime.now()
        formatted_time = now.strftime('%Y-%m-%d_%H-%M-%S')
        video_out_file = formatted_time+f"_skyreel_{args.width}-{args.num_frames}f_cfg-{args.guidance_scale}_steps-{args.num_inference_steps}_seed-{args.seed}_{idx}.mp4"
        save_video_with_quality(output, f"{out_dir}/{video_out_file}", args.fps, args.mbps)
