from ..gradio.diffusors import canny, scribble, depth

diffusers = {
    "canny": canny,
    "scribble": scribble,
    "depth": depth,
}
def main(controlnet_type, diffuser_params):
   diffuser = diffusers[controlnet_type]
   return diffuser.process(**diffuser_params)
