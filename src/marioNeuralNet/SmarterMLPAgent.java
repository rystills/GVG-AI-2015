package marioNeuralNet;

import java.util.ArrayList;

import core.game.StateObservation;
import core.player.AbstractPlayer;
import marioNeuralNet.Agent;
import marioNeuralNet.Evolvable;
import marioNeuralNet.SmarterMLP;
import ontology.Types.ACTIONS;
import tools.ElapsedCpuTimer;

public class SmarterMLPAgent extends AbstractPlayer implements Evolvable {

    public SmarterMLP mlp;
    private String name = "SmarterMLPAgent";
    //number of input nodes can be toggled; output nodes should remain at 6 (6 potential actions)
    int numberOfOutputs = 0;
    final int numberOfInputs = 10;

    /**
     * construct a new SmarterMLPAgent with the specified underlying MLP
     * @param mlp: the mlp to use internally for this agent
     */
    private SmarterMLPAgent(SmarterMLP mlp) {
        this.mlp = mlp;
    }
    
    /**
     * construct a new SmarterMLPAgent with the default mlp configuration (10 inputs, hidden layer nodes, and outputs)
     */
    public SmarterMLPAgent(StateObservation gameState, ElapsedCpuTimer elapsedTimer) {
    	//check first run
    	if (numberOfOutputs == 0) {
    		numberOfOutputs = gameState.getAvailableActions().size();
    		mlp = new SmarterMLP(numberOfInputs, numberOfOutputs, numberOfOutputs);
    	}
    }
    
    //satisfy Agent implementation
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public void giveIntermediateReward(float intermediateReward){ }
    public void reset() { mlp.reset(); }

    //satisfy Evolvable implementation
    public Evolvable getNewInstance() { return new SmarterMLPAgent(mlp.getNewInstance()); }
    public Evolvable copy() { return new SmarterMLPAgent(mlp.copy()); }

    /**
     * integrate basic stateful data from the specified environment instance
     * @param environment: the currently running environment instance from which we wish to integrate state information
     */
    public void integrateObservation() {}

    /**
     * mutate the hidden layer of our NN using the last set mutation magnitude
     */
    public void mutate() { mlp.mutate(); }
    
    /**
     * set the mutation magnitude and mutate the hidden layer of our NN
     * @param mutationMagnitude: the magnitude to apply to our mutation
     */
    public void mutate(float mutationMagnitude) {
    	mlp.mutate(mutationMagnitude);
    }
    
    public void recombine(SmarterMLPAgent parent1, SmarterMLPAgent parent2) {
    	this.mlp.psoRecombine(this.mlp, parent1.mlp, parent2.mlp);
    }
	
    /**
	 * check our inputs, propagate them, and return some output
	 * @return the array of keypresses comprising the action our agent wishes to perform
	 */
	@Override
	public ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
		//construct our input layer from each of our input conditions
    	double[] inputs = new double[] {
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1
    	};
    	
    	//construct our output layer by propagating our hidden layer from our inputs
        double[] outputs = mlp.propagate(inputs);
        
        //find largest output and use that as our action
        double largestVal = 0;
        int largestOutput = 0;
        ArrayList<ACTIONS> act = stateObs.getAvailableActions();
        for (int i = 0; i < outputs.length; ++i) {
        	if (outputs[i] > largestVal) {
        		largestVal = outputs[i];
        		largestOutput = i;
        	}
        }
        return act.get(largestOutput);
	}
}