# Residual-Encoding-for-Domain-Specific-Video

We built a streaming video codec which integrates Convolutional Neural Network autoencoder with H.264 to achieve a higher compression ratio than using conventional H.264 codec.
Because it's difficult to train high-resolution video using CNN, we further used gradient-checkpointing to fit large models in GPU.

## Steps
---Please put the video(raw and compressed) in the data folder.---

To run the program, first specify the required parameters, including 
"height", "width", "C", and "path" in "main.py", "encoder.py" and "decoder.py".
Note that "height", "width" in "decoder.py" denotes the compressed size.

We use the following example to run the program.
When the path of raw and low-quality(compressed) videos are "./data/NBA_raw.mp4" and 
"./data/NBA_com.mp4" respectively, we let "path = ./data/NBA"

1. To train the autoencoder, use python main.py.
    The trained model will be saved in the folder "./saved_model".
2. To encode video, use python encoder.py.
    The encoder will load model from "./saved_model" folder, huffman codec will
    be saved in "./saved_model" folder, and the encoded residual will be 
    saved as "./data/NBA_residual.npy".
3. To decode video, use python decoder.py.
    The decoder will load model and residual, and the 
    reconstructed video will be saved as "NBA_decoded.avi".

## Reference: 
[1]Tsai, Y.H., Liu, M.Y., Sun, D., Yang, M.H., Kautz, J.: Learning binary residual representations for domain-specific video streaming. In: AAAI

[2]subpixel: A subpixel convolutional neural network implementation with Tensorflow, https://github.com/tetrachrome/subpixel
