import gradio as gr
import argparse
import sys
import time
import os
import random
sys.path.append("..")
from skyreelsinfer import TaskType
from skyreelsinfer.offload import OffloadConfig
from skyreelsinfer.skyreels_video_infer import SkyReelsVideoInfer
from diffusers.utils import export_to_video
from diffusers.utils import load_image

predictor = None
current_task_type = "t2v"  # 默认模式为 t2v

def get_transformer_model_id(task_type: str) -> str:
    return "Skywork/SkyReels-V1-Hunyuan-I2V" if task_type == "i2v" else "Skywork/SkyReels-V1-Hunyuan-T2V"

def init_predictor(task_type: str, gpu_num: int = 1):
    global predictor
    predictor = SkyReelsVideoInfer(
        task_type=TaskType.I2V if task_type == "i2v" else TaskType.T2V,
        model_id=get_transformer_model_id(task_type),
        quant_model=True,
        world_size=gpu_num,
        is_offload=True,
        offload_config=OffloadConfig(
            high_cpu_memory=True,
            parameters_level=True,
            compiler_transformer=False,
        )
    )

def generate_video(prompt, seed, width, height, num_frames=97, image=None):
    global current_task_type
    print(f"task_type: {current_task_type}, image: {type(image)}")

    if seed == -1:
        random.seed(time.time())
        seed = int(random.randrange(4294967294))

    kwargs = {
        "prompt": prompt,
        "height": int(height),
        "width": int(width),
        "num_frames": int(num_frames),  # 使用用户输入的帧数
        "num_inference_steps": 30,
        "seed": seed,
        "guidance_scale": 6.0,
        "embedded_guidance_scale": 1.0,
        "negative_prompt": "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
        "cfg_for": False,
    }

    if current_task_type == "i2v":
        assert image is not None, "please input image"
        kwargs["image"] = load_image(image=image)
    
    global predictor
    if predictor is None or predictor.task_type != (TaskType.I2V if current_task_type == "i2v" else TaskType.T2V):
        init_predictor(current_task_type)
    output = predictor.inference(kwargs)
    save_dir = f"./result/{current_task_type}"
    os.makedirs(save_dir, exist_ok=True)
    video_out_file = f"{save_dir}/{prompt[:100].replace('/','')}_{seed}.mp4"
    print(f"generate video, local path: {video_out_file}")
    export_to_video(output, video_out_file, fps=24)
    return video_out_file, kwargs

def set_task_type(task_type, t2v_btn, i2v_btn):
    global current_task_type
    current_task_type = task_type
    if task_type == "t2v":
        return gr.update(variant="primary"), gr.update(variant="secondary"), gr.update(visible=False), gr.update(visible=True)
    else:  # i2v
        return gr.update(variant="secondary"), gr.update(variant="primary"), gr.update(visible=True), gr.update(visible=False)

def create_gradio_interface():
    """Create a Gradio interface with Input Prompt on its own row, followed by Generate, T2V, and I2V buttons on one row."""
    with gr.Blocks(theme=gr.themes.Default()) as demo:
        # 自定义 CSS
        css = """
        .gr-image {
            max-width: 480px !important;
            max-height: 848px !important;
            object-fit: contain !important;
        }
        .gr-video {
            max-width: 480px !important;
            max-height: 848px !important;
            object-fit: contain !important;
        }
        .gr-button {
            height: 100% !important;
        }
        """
        demo.css = css

        # 顶部：动态显示图像上传或占位文本
        with gr.Row() as top_row:
            with gr.Column(scale=1, visible=True) as image_col:
                image = gr.Image(label="Upload Image", type="filepath", height=848, width=480, visible=False)
                placeholder = gr.Markdown("### Text-to-Video Mode\nNo image required.", visible=True)
            with gr.Column(scale=2):
                output_video = gr.Video(label="Generated Video", height=848, width=480)

        # 中间：参数调整区域
        with gr.Column():
            # "Input Prompt" 单独一行
            with gr.Row():
                with gr.Column(scale=4):  # 扩大宽度以占据整行
                    prompt = gr.Textbox(label="Input Prompt", lines=1)
            
            # "Generate Video", "T2V", 和 "I2V" 按钮在一行
            with gr.Row():
                with gr.Column(scale=1, min_width=120):
                    submit_button = gr.Button("Generate Video", variant="primary")
                with gr.Column(scale=1, min_width=100):
                    t2v_button = gr.Button("T2V", variant="primary")  # 默认选中 t2v
                with gr.Column(scale=1, min_width=100):
                    i2v_button = gr.Button("I2V", variant="secondary")
            
            # 参数区域
            with gr.Row():
                with gr.Column(scale=1):
                    seed = gr.Number(label="Random Seed", value=-1, precision=0)
                with gr.Column(scale=1):
                    num_frames = gr.Number(label="Num Frames", value=97, precision=0)
                with gr.Column(scale=1):
                    width = gr.Number(label="Width", value=544, precision=0)
                with gr.Column(scale=1):
                    height = gr.Number(label="Height", value=960, precision=0)
        
        # 底部：输出参数
        with gr.Column():
            output_params = gr.Textbox(label="Output Parameters")

        # 绑定任务类型切换逻辑
        t2v_button.click(
            fn=lambda: set_task_type("t2v", t2v_button, i2v_button),
            inputs=[],
            outputs=[t2v_button, i2v_button, image, placeholder]
        )
        i2v_button.click(
            fn=lambda: set_task_type("i2v", t2v_button, i2v_button),
            inputs=[],
            outputs=[t2v_button, i2v_button, image, placeholder]
        )

        # 提交按钮逻辑
        submit_button.click(
            fn=generate_video,
            inputs=[prompt, seed, width, height, num_frames, image],
            outputs=[output_video, output_params],
        )

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gradio app.")
    parser.add_argument("--gpu_num", type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()
    
    demo = create_gradio_interface()
    demo.launch()