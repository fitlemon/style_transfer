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
        with torch.no_grad():
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

    def _encode_image_to_base64(self, img):
        """Convert PIL image to base64 encoded string"""
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_base64}"

    def generate_content_description(self, img) -> str:
        """
        Generate a textual description of the content image, focusing on poses, objects, and people.
        """
        # Convert the image to a base64-encoded JPEG
        data_url = self._encode_image_to_base64(img)

        try:
            # Import and initialize OpenAI client
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Create a chat completion request focused on content description
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image focusing specifically on the exact poses, positions of people, "
                                "the objects present, and their spatial arrangement. Include details about the people's positions, "
                                "expressions, and clothing. Be precise about the composition, but don't describe the style or artistic qualities.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                max_tokens=250,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating content description with OpenAI: {e}")
            return "A photograph showing people and objects in a natural arrangement."

    def generate_style_description(self, img) -> str:
        """
        Generate a textual description of the style image, focusing on texture, style, and theme.
        """
        # Convert the image to a base64-encoded JPEG
        data_url = self._encode_image_to_base64(img)

        try:
            # Import and initialize OpenAI client
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Create a chat completion request focused on style description
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe the artistic style, texture, color palette, and visual theme of this image. "
                                "Focus on how it's rendered (digital art, oil painting, photography style, etc.), "
                                "the mood it conveys, lighting techniques, and any distinctive visual characteristics. "
                                "Don't describe the content or subjects of the image.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                max_tokens=200,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating style description with OpenAI: {e}")
            return "A visually striking artistic style with vibrant colors and detailed textures."

    def combine_prompts(self, content_description: str, style_description: str) -> str:
        """
        Combine content and style descriptions into a coherent prompt for image generation.
        """
        try:
            # Import and initialize OpenAI client
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Create a chat completion request to combine descriptions
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a prompt engineering assistant. Your task is to create effective "
                        "Stable Diffusion prompts by combining content and style descriptions.",
                    },
                    {
                        "role": "user",
                        "content": f"I need to create a high-quality image using Stable Diffusion. Please combine these two descriptions into a single, coherent prompt with 2 short sentences (maximum 77 tokens):\n\n"
                        f"CONTENT DESCRIPTION (preserve all positioning, people, and objects exactly as described):\n{content_description}\n\n"
                        f"STYLE DESCRIPTION (apply this artistic style to the content):\n{style_description}\n\n"
                        f"Create a prompt that would generate an image with the exact content described (same people, poses, objects) "
                        f"but rendered in the artistic style described. The prompt should be concise but detailed, focusing on both content and style aspects.",
                    },
                ],
                max_tokens=300,
            )
            combined_prompt = response.choices[0].message.content
        except Exception as e:
            print(f"Error combining prompts with OpenAI: {e}")
            combined_prompt = f"A detailed image showing {content_description} rendered in {style_description} style."

        # Prepend additional style tokens to the prompt
        full_prompt = f"(photorealistic:1.2), raw, masterpiece, high quality, 8k, {combined_prompt}"
        return full_prompt

    def generate_prompt(self, content_img, style_img=None):
        """
        Generate a textual prompt for image synthesis.
        If only content_img is provided, generates a basic description.
        If both images are provided, combines content and style descriptions.
        """
        if style_img is None:
            # Legacy behavior - just describe the content
            return self._legacy_generate_prompt(content_img)
        else:
            # New behavior - combine content and style descriptions
            content_description = self.generate_content_description(content_img)
            style_description = self.generate_style_description(style_img)
            return self.combine_prompts(content_description, style_description)

    def _legacy_generate_prompt(self, img) -> str:
        """
        Legacy method for generating a basic prompt from a single image.
        """
        # Convert the image to a base64-encoded JPEG
        data_url = self._encode_image_to_base64(img)

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
                                "text": "Describe this photo in 2 short sentences as a prompt for generating an image in Stable Diffusion, adding vivid details (e.g. color, clothes, objects around, etc.) to help the model understand the image better.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                    "detail": "low",
                                },
                            },
                        ],
                    },
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

    def cleanup(self):
        """Release GPU resources when done with a generation"""
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Force Python's garbage collector
        import gc

        gc.collect()
