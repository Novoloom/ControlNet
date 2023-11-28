import depth
import canny
import scribble

diffusers = {
    "canny": canny,
    "scribble": scribble,
    "depth": depth,
}
def main(controlnet_type, diffuser_params):
   diffuser = diffusers[controlnet_type]
   return diffuser.process(**diffuser_params)
