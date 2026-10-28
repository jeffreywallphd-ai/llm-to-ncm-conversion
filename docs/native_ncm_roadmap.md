# Future native-NCM conversion (outside current studies)

Current code retains an AR target verifier. A native NCM generator would require a
trained multiscale text codec (e.g., residual VQ), discrete latent supervision,
coarse-to-fine latent generator training, and a text decoder that reconstructs useful
outputs. The current continuous query groups are not established semantic concepts.

First evaluate codec reconstruction, rate/distortion, level semantics and failure
modes on licensed held-out data. Then train latent prediction and text reconstruction
with explicit data/compute budgets, compare quality and measured end-to-end generation
cost against strong AR/diffusion/NCM controls, and assess distribution/quality changes.
The frozen-target exact-verification guarantee does not automatically transfer.

No native conversion experiment or native text-codec checkpoint is included. The current auxiliary prediction-state codec is documented separately in autoencoder_architecture.md. This requires a
new scientific plan, budget, quality endpoints and contribution audit after the
present technical and financial-workflow questions are evaluated.
