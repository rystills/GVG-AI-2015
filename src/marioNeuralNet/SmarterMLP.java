package ch.idsia.evolution;

import java.util.Random;

public class SmarterMLP implements FA<double[], double[]>, Evolvable {

	public static final Random random = new Random();
    private double[][] firstConnectionLayer;
    private double[][] secondConnectionLayer;
    private double[] hiddenNeurons; //hidden layer
    private double[] outputs; //output layer
    private double[] inputs; //input layer
    public double mutationMagnitude = 0.1; //severity multiplier applied to all mutations

    //initial values
    public static double mean = 0.0f; //initial mean
    public static double deviation = 0.1f; //initial deviation

    /**
     * construct a new SmarterMLP instance to handle mutation and recombination
     * @param numberOfInputs: the number of inputs with which to build our NN
     * @param numberOfHidden: the number of hidden nodes supported by our NN
     * @param numberOfOutputs: the number of outputs for our NN (set to 6 to match number of game actions)
     */
    public SmarterMLP(int numberOfInputs, int numberOfHidden, int numberOfOutputs) {
        firstConnectionLayer = new double[numberOfInputs][numberOfHidden];
        secondConnectionLayer = new double[numberOfHidden][numberOfOutputs];
        hiddenNeurons = new double[numberOfHidden];
        outputs = new double[numberOfOutputs];
        inputs = new double[numberOfInputs];
        initializeLayer(firstConnectionLayer);
        initializeLayer(secondConnectionLayer);
    }

    /**
     * construct a new SmarterMLP instance given existing connection layers
     * @param firstConnectionLayer: the first internal connection layer
     * @param secondConnectionLayer: the second internal connection layer
     * @param numberOfHidden: the number of hidden nodes supported by our NN
     * @param numberOfOutputs: the number of outputs for our NN (set to 6 to match number of game actions)
     */
    public SmarterMLP(double[][] firstConnectionLayer, double[][] secondConnectionLayer, int numberOfHidden, int numberOfOutputs) {
        this.firstConnectionLayer = firstConnectionLayer;
        this.secondConnectionLayer = secondConnectionLayer;
        inputs = new double[firstConnectionLayer.length];
        hiddenNeurons = new double[numberOfHidden];
        outputs = new double[numberOfOutputs];
    }
    
    //satisfy Evolvable implementation
    public void reset() {}
    public SmarterMLP getNewInstance() { return new SmarterMLP(firstConnectionLayer.length, secondConnectionLayer.length, outputs.length); }
    public SmarterMLP copy() {
        SmarterMLP copy = new SmarterMLP(copy(firstConnectionLayer), copy(secondConnectionLayer), hiddenNeurons.length, outputs.length);
        copy.setMutationMagnitude(mutationMagnitude);
        return copy;
    }
    
    //satisfy FA implementation
    public double[] approximate(double[] doubles) { return propagate(doubles); }
    
    //getters and setters for mutation magnitude
    public double getMutationMagnitude() { return mutationMagnitude; }
    public void setMutationMagnitude(double mutationMagnitude) { this.mutationMagnitude = mutationMagnitude; }

    /**
     * initialize the specified layer with random values augmented by our starting deviation and mean
     * @param layer: the layer to initialize
     */
    protected void initializeLayer(double[][] layer) {
        for (int i = 0; i < layer.length; i++) {
            for (int j = 0; j < layer[i].length; j++) {
                layer[i][j] = (random.nextGaussian() * deviation + mean);
            }
        }
    }

    /**
     * deepcopy the input 2darray
     * @param original: the 2darray to copy
     * @return the newly created array deepcopy
     */
    private double[][] copy(double[][] original) {
        double[][] copy = new double[original.length][original[0].length];
        for (int i = 0; i < original.length; System.arraycopy(original[i], 0, copy[i], 0, original[i].length), ++i);
        return copy;
    }
    
    /**
     * mutate our internal layers using the current mutation magnitude
     */
    public void mutate() {
    	mutate(firstConnectionLayer);
        mutate(secondConnectionLayer);
    }

    /**
     * mutate our internal layers using the specified mutation magnitude
     * @param newMutationMagnitude: the new mutation magnitude to apply and use
     */
    public void mutate(float newMutationMagnitude) {
    	if (newMutationMagnitude != -1) {
    		setMutationMagnitude(newMutationMagnitude);	
    	}
    	mutate();
    }

    /**
     * mutate the 2darray corresponding to the input NN layer
     * @param array: the 2darray to mutate
     */
    private void mutate(double[][] array) {
        for (double[] anArray : array) {
            mutate(anArray);
        }
    }
    
    /**
     * mutate a single dimension of our 2darray
     * @param array: the single dimensional array to mutate
     */
    private void mutate(double[] array) {
        for (int i = 0; i < array.length; array[i] += random.nextGaussian() * mutationMagnitude, ++i);
    }

    /**
     * recombine this NN's internal layers from the specified parent NNs, using the psoRecombination algorithm
     * @param last: the starting NN
     * @param pBest: the first elite parent
     * @param gBest: the second elite parent
     */
    public void psoRecombine(SmarterMLP last, SmarterMLP pBest, SmarterMLP gBest) {
        final double ki = 0.729844;
        final double phi = 2.05;

        double phi1 = phi * random.nextDouble();
        double phi2 = phi * random.nextDouble();
                
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < hiddenNeurons.length; j++) {
                recombineLayerIndices(firstConnectionLayer, last.firstConnectionLayer, 
                		pBest.firstConnectionLayer, gBest.firstConnectionLayer, ki, phi1, phi2, i, j);
            }
        }

        for (int i = 0; i < hiddenNeurons.length; i++) {
            for (int j = 0; j < outputs.length; j++) {
            	recombineLayerIndices(secondConnectionLayer, last.secondConnectionLayer, 
                		pBest.secondConnectionLayer, gBest.secondConnectionLayer, ki, phi1, phi2, i, j);
            }
        }

    }
    
    /**
     * recombine the specified indices of the desired layer, using the specified inputs
     * @param ourLayer: the layer (first or second) on which to perform recombination
     * @param lastLayer: the starting NN layer
     * @param pBestLayer: the first parent NN layer
     * @param gBestLayer: the second parent NN layer
     * @param ki: offset magnitude
     * @param phi1: first random offset applied to combination constant phi
     * @param phi2: second random offset applied to combination constant phi
     * @param i: first index
     * @param j: second index
     */
    public void recombineLayerIndices(double[][] ourLayer, double[][]lastLayer, double[][]pBestLayer, 
    		double[][]gBestLayer, double ki, double phi1, double phi2, int i, int j) {
    	ourLayer[i][j] = (double) (ourLayer[i][j] + ki * (ourLayer[i][j] 
    		- ((double[][]) (lastLayer))[i][j]
    		+ phi1 * (((double[][]) (pBestLayer))[i][j] - ourLayer[i][j])
    		+ phi2 * (((double[][]) (gBestLayer))[i][j] - ourLayer[i][j]))); 
    }

    /**
     * propagate the specified inputs through our NN, returning the resulting outputs
     * @param inputIn: the input data we collected
     * @return the output values calculated by our NN
     */
    public double[] propagate(double[] inputIn) {
    	//update inputs array to match our most recent input probe
        System.arraycopy(inputIn, 0, this.inputs, 0, inputIn.length);
        
        //propogate first connection layer and clamp results
        propagateOneStep(inputs, hiddenNeurons, firstConnectionLayer);
        tanh(hiddenNeurons);
        //propogate second connection layer and clamp results
        propagateOneStep(hiddenNeurons, outputs, secondConnectionLayer);
        tanh(outputs);

        return outputs;
    }

    /**
     * propagate the values from fromLayer over to toLayer by a factor of connections
     * @param fromLayer: the initial layer
     * @param toLayer: the destination layer
     * @param connections: the connections defining the relation between fromLayer and toLayer
     */
    private void propagateOneStep(double[] fromLayer, double[] toLayer, double[][] connections) {
    	//clear toLayer
    	for (int i = 0; i < toLayer.length; toLayer[i] = 0, ++i);
    	
    	//copy from fromLayer over to toLayer
        for (int from = 0; from < fromLayer.length; from++) {
            for (int to = 0; to < toLayer.length; to++) {
                toLayer[to] += fromLayer[from] * connections[from][to];
            }
        }
    }
    
    /**
     * calculate tanh for all values in the specified array, clamping them between +-1
     * @param array: the array to use for calculations
     */
    private void tanh(double[] array) {
        for (int i = 0; i < array.length; array[i] = Math.tanh(array[i]), ++i);
    }
}