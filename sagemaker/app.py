from dalle_model import DalleModel
import jax
import time

if __name__ == "__main__":
    print("<<< devices", jax.devices())

    dalle_model = DalleModel("Mini")

    print ("start!")
    start = time.time()
    a = dalle_model.generate_images("warm-up", 1)
    end = time.time()
    print ("time",end-start)

    print ("result: ", a)