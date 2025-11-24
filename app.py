import gradio as gr
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import matplotlib
from ppd.models.ppd import PixelPerfectDepth
from ppd.utils.set_seed import set_seed
from ppd.utils.timesteps import Timesteps
import tempfile
import zipfile

# Global variables
model = None
current_device = None

def get_available_devices():
    devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            devices.append(f"cuda:{i} ({device_name})")
    elif torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices

def load_model(device_str, sampling_steps):
    global model, current_device
    
    # Parse device string (e.g., "cuda:0 (RTX 3090)" -> "cuda:0")
    if "cuda" in device_str:
        device_id = device_str.split(" ")[0]
        device = torch.device(device_id)
    elif "mps" in device_str:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Check if we need to reload (device changed)
    if model is not None and current_device != device:
        print(f"Device changed from {current_device} to {device}. Reloading model...")
        del model
        torch.cuda.empty_cache()
        model = None
        current_device = None

    if model is None:
        print(f"Loading model on {device}...")
        # Paths are relative to the container/workspace root
        semantics_pth = 'checkpoints/depth_anything_v2_vitl.pth'
        ppd_pth = 'checkpoints/ppd.pth'
        
        if not os.path.exists(semantics_pth) or not os.path.exists(ppd_pth):
            raise FileNotFoundError(f"Checkpoints not found. Please ensure {semantics_pth} and {ppd_pth} exist.")

        model = PixelPerfectDepth(semantics_pth=semantics_pth, sampling_steps=sampling_steps)
        model.load_state_dict(torch.load(ppd_pth, map_location='cpu'), strict=False)
        model = model.to(device).eval()
        
        # Update internal device and timesteps
        model.device = device
        model.sampling_timesteps = Timesteps(
            T=model.schedule.T,
            steps=model.sampling_steps,
            device=device,
        )
        model.sampler.timesteps = model.sampling_timesteps
        
        current_device = device
    else:
        # Check if we need to update sampling steps
        if hasattr(model, 'sampling_steps') and model.sampling_steps != sampling_steps:
            model.sampling_steps = sampling_steps
            # Re-init timesteps if steps changed
            model.sampling_timesteps = Timesteps(
                T=model.schedule.T,
                steps=model.sampling_steps,
                device=model.device,
            )
            model.sampler.timesteps = model.sampling_timesteps
            
    return model, device

def infer_core(model_instance, image_bgr):
    H, W = image_bgr.shape[:2]
    with torch.no_grad():
        depth, _ = model_instance.infer_image(image_bgr)
        depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)[0, 0]
        depth = depth.squeeze().cpu().numpy()
    return depth

def colorize_depth(depth):
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    vis_depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    vis_depth = vis_depth.astype(np.uint8)
    vis_depth = (cmap(vis_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    return vis_depth

def predict(image, device_choice, sampling_steps, input_size, save_npy):
    if image is None:
        return None, None

    set_seed(666)
    
    try:
        model_instance, device = load_model(device_choice, sampling_steps)
    except Exception as e:
        raise gr.Error(f"Error loading model: {str(e)}")

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        depth = infer_core(model_instance, image_bgr)
    except Exception as e:
        raise gr.Error(f"Error during inference: {str(e)}")
    finally:
        torch.cuda.empty_cache()
        
    vis_depth = colorize_depth(depth)
    
    # Convert BGR visualization back to RGB for Gradio
    vis_depth_rgb = cv2.cvtColor(vis_depth, cv2.COLOR_BGR2RGB)

    npy_path = None
    if save_npy:
        temp_dir = tempfile.mkdtemp()
        npy_path = os.path.join(temp_dir, 'prediction.npy')
        np.save(npy_path, depth)

    return vis_depth_rgb, npy_path

def process_batch(files, device_choice, sampling_steps, save_npy):
    if not files:
        return None
        
    set_seed(666)
    
    try:
        model_instance, device = load_model(device_choice, sampling_steps)
    except Exception as e:
        return f"Error loading model: {str(e)}"
        
    out_tmp_dir = tempfile.mkdtemp()
    vis_dir = os.path.join(out_tmp_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    if save_npy:
        npy_dir = os.path.join(out_tmp_dir, "npy")
        os.makedirs(npy_dir, exist_ok=True)
        
    processed_count = 0
    
    for file_obj in files:
        # Gradio passes file objects or paths depending on version/config
        # In recent Gradio, file_count="directory" returns list of file paths (temp files)
        # file_obj is likely a NamedString or similar wrapper, or just a path string
        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        
        filename = os.path.basename(file_path)
        # Skip non-image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            continue
            
        try:
            image = cv2.imread(file_path)
            if image is None: continue
            
            depth = infer_core(model_instance, image)
            vis_depth = colorize_depth(depth)
            
            # Save visualization
            cv2.imwrite(os.path.join(vis_dir, os.path.splitext(filename)[0] + '.png'), vis_depth)
            
            if save_npy:
                np.save(os.path.join(npy_dir, os.path.splitext(filename)[0] + '.npy'), depth)
                
            processed_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    if processed_count == 0:
        return None

    # Create ZIP
    zip_path = os.path.join(out_tmp_dir, "results.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(vis_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.join("vis", file))
        if save_npy:
            for root, dirs, files in os.walk(npy_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.join("npy", file))
                    
    return zip_path

# Gradio Interface
with gr.Blocks(title="Pixel Perfect 深度估计") as demo:
    gr.Markdown("# Pixel Perfect 深度估计")
    gr.Markdown("上传图片以使用 Pixel Perfect Depth 模型进行深度估计。")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 设置")
            device_list = get_available_devices()
            device_dropdown = gr.Dropdown(choices=device_list, value=device_list[0], label="设备")
            sampling_steps = gr.Slider(minimum=1, maximum=10, value=4, step=1, label="采样步数")
            save_npy_checkbox = gr.Checkbox(label="保存 NPY 文件", value=False)

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("单张图片"):
                    with gr.Row():
                        with gr.Column():
                            input_image = gr.Image(label="输入图片", type="numpy")
                            run_btn = gr.Button("运行深度估计", variant="primary")
                        with gr.Column():
                            output_image = gr.Image(label="深度图", format="png")
                            output_file = gr.File(label="下载 NPY 文件", visible=True)
                            
                    run_btn.click(
                        fn=predict,
                        inputs=[input_image, device_dropdown, sampling_steps, gr.State(None), save_npy_checkbox],
                        outputs=[output_image, output_file]
                    )
                    
                with gr.TabItem("批量处理 (文件夹)"):
                    gr.Markdown("上传包含图片的文件夹。处理完成后将提供 ZIP 包下载。")
                    with gr.Row():
                        with gr.Column():
                            # file_count="directory" allows uploading a folder
                            input_folder = gr.File(file_count="directory", label="上传文件夹")
                            batch_run_btn = gr.Button("批量运行", variant="primary")
                        with gr.Column():
                            output_zip = gr.File(label="下载结果 (ZIP)")
                            
                    batch_run_btn.click(
                        fn=process_batch,
                        inputs=[input_folder, device_dropdown, sampling_steps, save_npy_checkbox],
                        outputs=[output_zip]
                    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
