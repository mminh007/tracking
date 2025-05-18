import os
import sys
import subprocess
import cv2
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
from tqdm import tqdm

# ======= Zero-DCE: Load model from repo =======
os.chdir(os.getcwd())

def clone_repo(url):
    try:
        subprocess.run(["git", "clone", url], check=True)
        print(f"✅ Cloned: {url}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone: {url}\n{e}")



def main(args):

    clone_repo("https://github.com/Li-Chongyi/Zero-DCE.git")
    clone_repo("https://github.com/JingyunLiang/SwinIR.git")

    sys.path.append("Zero-DCE")
    zero_dce_code = os.path.join("Zero-DCE", "Zero-DCE_code")
    sys.path.append(zero_dce_code)

    import model 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    zero_dce = model.enhance_net_nopool().to(device)
    zero_dce.load_state_dict(torch.load("/content/Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth", map_location=device))
    zero_dce.eval()

    # ======= SwinIR: super-resolution =======
    sys.path.append('SwinIR')
    from models.network_swinir import SwinIR

    swinir = SwinIR(
        upscale=2,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffledirect',
        resi_connection='1conv'
    ).to(device)

    state_dict = torch.hub.load_state_dict_from_url(
        url=args.swinir_weight,
        map_location=device
    )

    # swinir = SwinIR(
    #     upscale=2,
    #     in_chans=3,
    #     img_size=64,
    #     window_size=8,
    #     img_range=1.,
    #     depths=[6, 6, 6, 6, 6, 6],
    #     embed_dim=180,
    #     num_heads=[6, 6, 6, 6, 6, 6],
    #     mlp_ratio=2,
    #     upsampler='pixelshuffle',
    #     resi_connection='1conv'
    # ).to(device)

    # state_dict = torch.hub.load_state_dict_from_url(
    #     url='https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth',
    #     map_location=device)

    swinir.load_state_dict(state_dict['params'], strict=True)
    swinir.eval()

    # ======= Folder setup =======
    input_folder = args.input_folder
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # ======= Loop over images =======
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

    for filename in tqdm(image_files, desc="Enhancing low-light + resolution"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # Load and resize image
            img = cv2.imread(input_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # ======= (1) Enhance Low-Light by Zero-DCE =======
            img_pil = Image.fromarray(img)
            img_tensor = to_tensor(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                #enhanced = zero_dce_enhance(zero_dce, img_tensor).clamp(0, 1)
                _,enhanced,_ = zero_dce(img_tensor)

            del img_tensor
            torch.cuda.empty_cache()
            # ======= (2) Super-Resolution by SwinIR =======
            with torch.no_grad():
                out_tensor = swinir(enhanced).clamp_(0, 1)

            del enhanced
            torch.cuda.empty_cache()

            out_img = to_pil_image(out_tensor.squeeze(0).cpu())
            out_img.save(output_path)

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print("✅ Enhancing low-light + resolution Done!.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhance images using Zero-DCE and SwinIR")
    parser.add_argument("--input", type=str, required=True, help="Path to input folder")
    parser.add_argument("--output", type=str, required=True, help="Path to output folder")
    parser.add_argument("--swinir-weight", type=str, default='https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth', help="Path to SwinIR weight file")
    args = parser.parse_args()

    main(args)