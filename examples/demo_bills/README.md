# Example bills (new, not used in training)

Ten artificial Indian-style bills for trying the app: `bill_01.jpg` ... `bill_10.jpg`, each with its exact label in `bill_NN.json`.

- Made by `splitsnap/synth.py` with seeds 7001-7010 and then given the `synth` photo augmentation (background, tilt, shadow, fading, JPEG).
  The training and test data used other seeds (1000/2000/3000), so these exact bills were never seen by the model.
- They are **generated, not real photos**: the layouts and wording come from the same generator the model trained on, so reading them
  is easier than a real restaurant bill. They show the flow, not real-world accuracy.
- The app lists them on its first screen ("Or try an example photo"); each one goes through the real model like a normal upload.
- Measured with the real model (2026-10-09): all 10 came back with the right number of items, the right number of charges and the right
  grand total. Bills 04 and 05 are faded and trigger the app's "retake" prompt (you can continue anyway).
