from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector
from diffusers.utils import load_image
import torch
import base64
import io
import os
from dotenv import load_dotenv

# Load OpenAI API key
load_dotenv()


class StyleTransferModel:
    def __init__(self, ip_adapter_scale: float = 0.5, device: str = "cuda"):
        # Check if CUDA is available, otherwise use CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Store the IP adapter scale as a class attribute
        self.ip_adapter_scale = ip_adapter_scale

        # Load the ControlNet model for canny detection
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        # Load the Stable Diffusion pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "Yntec/AbsoluteReality",
            controlnet=self.controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        # Load the IP Adapter weights
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin"
        )

        # Enable CPU offloading if using CUDA
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

        # Set initial IP adapter scale
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)

        # Initialize the Canny detector
        self.canny = CannyDetector()

    # Add a method to update the IP adapter scale
    def set_ip_adapter_scale(self, scale: float):
        """Set the IP adapter scale and store the value"""
        self.ip_adapter_scale = scale
        self.pipe.set_ip_adapter_scale(scale)

    def preprocess_image(self, image_path: str, size: tuple = (512, 512)):
        """
        Load an image from a path and resize it.
        """
        img = load_image(image_path)
        img.thumbnail(size)
        return img

    def get_canny_image(self, img, detect_resolution: int = 512):
        """
        Compute the Canny edge image from the input.
        """
        return self.canny(
            img, detect_resolution=detect_resolution, image_resolution=img.size[1]
        )

    def generate(
        self,
        prompt: str,
        base_img,
        ip_adapter_img,
        canny_img,
        guidance_scale: float = 6,
        conditioning_scale: float = 0.7,
        inference_steps: int = 20,
        num_images: int = 1,
    ):
        """
        Generate images using the Stable Diffusion pipeline.
        """
        outputs = self.pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, distorted, disfigured, bad anatomy",
            height=base_img.size[1],
            width=base_img.size[0],
            ip_adapter_image=ip_adapter_img,
            image=canny_img,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=conditioning_scale,
            num_inference_steps=inference_steps,
            num_images_per_prompt=num_images,
        )
        return outputs.images

    def generate_prompt(self, img) -> str:
        """
        Generate a textual prompt from an image using the OpenAI API.
        """
        # Convert the image to a base64-encoded JPEG
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_base64}"

        try:
            # Import and initialize OpenAI client
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Create a chat completion request.
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this photograph in 2 short sentences as a prompt for generating an image in Stable Diffusion, adding vivid details (e.g. color, clothes, objects around, etc.) to help the model understand the image better. This photo is from kindergarten. Please be careful with the content. Describe only kid's activities. Add to prompt that the model should generate a happy and joyful image without NFSW.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                    {"role": "user", "content": data_url},
                ],
                max_tokens=300,
            )
            prompt = response.choices[0].message.content
        except Exception as e:
            print(f"Error generating prompt with OpenAI: {e}")
            # Fallback to a generic prompt if OpenAI API fails
            prompt = "A beautiful, detailed photograph with vivid colors and intricate details."

        # Prepend additional style tokens to the prompt
        full_prompt = (
            f"(photorealistic:1.2), raw, masterpiece, high quality, 8k, {prompt}"
        )
        return full_prompt
