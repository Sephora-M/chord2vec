# Chord2vec
The main goal of this work is to introduce techniques that can be used for learning high-quality embedding chord vectors from sequences of polyphonic music. 
We aim to achieve this by finding chord representations that are useful for predicting the neighboring chords in a musical piece. 

Please refer to the written report for information on notations, etc.

## Linear model  
This model assumes conditional independence between the notes in a the context chord c given a chord d:
<img src="http://www.sciweavers.org/tex2img.php?eq=p%28%5Cmathbf%7Bc%7D%20%3D%5Cmathbf%7Bc%7D%27%20%7C%20%5Cmathbf%7Bd%7D%29%20%3D%20%5Cprod_%7Bi%3D1%7D%5EN%20p%28c_i%20%3Dc_i%27%7C%20%5Cmathbf%7Bd%7D%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="p(\mathbf{c} =\mathbf{c}' | \mathbf{d}) = \prod_{i=1}^N p(c_i =c_i'| \mathbf{d})" width="235" height="53" />

## Autor regressive model
This model decomposes the context chord probability distribution according to the chain rule:
<img src="http://www.sciweavers.org/tex2img.php?eq=p%28%5Cmathbf%7Bc%7D%20%3D%5Cmathbf%7Bc%7D%27%20%7C%20%5Cmathbf%7Bd%7D%29%20%3D%20%5Cprod_%7Bi%3D1%7D%5EN%20p%28c_i%20%3Dc_i%27%7C%20%5Cmathbf%7Bd%7D%2C%20c_%7B%3Ci%7D%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="p(\mathbf{c} =\mathbf{c}' | \mathbf{d}) = \prod_{i=1}^N p(c_i =c_i'| \mathbf{d}, c_{<i})" width="267" height="53" ></i>

## Sequence to Sequence model
Sequence-to-sequence models allow to learn a mapping of input sequences of varying lengths (a chord) to output sequences also of varying lengths (a neighbor chord) . It uses a neural network architecture known as RNN Encoder-Decoder. 
The model estimates the conditional probability of a context chord c given an input chord d by first obtaining the fixed-length vector representation v of the input chord (given by the last state of the LSTM encoder) and then computing the probability of c with the LSTM decoder. 


