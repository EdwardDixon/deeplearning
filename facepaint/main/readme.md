###Facepaint
This is toy regression problem ("image regression", essentially a port of Karpathy's excellent ConvNetJS demo)[http://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html], written in Keras.  To run it, you'll need to have Python, Keras, and either TensorFlow or Theano (which Keras uses to do the actual computations).

What does it do?  You feed it a photo as *test data*; it then tries to learn to predict the (r,g,b) for a given (x,y).  

What's the point? Mostly, educating the programmer!  Because the output is so visual (it writes out one JPG per training iteration to show what it has learned so far), it really makes it easy to see the effect of changing the model (more/fewer layers, wider/narrower layers) or of tweaking meta-paramters or optimizers.  I lea
