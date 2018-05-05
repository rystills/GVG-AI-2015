package marioNeuralNet;

import java.util.ArrayList;
import java.util.Arrays;

import core.game.Observation;
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
    final int numberOfInputs = 7;

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
    
    public SmarterMLPAgent(int numOut) {
    	if (numberOfOutputs == 0) {
    		numberOfOutputs = numOut;
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
     * find the distance to the nearest observation type
     * @param stateObs our state observation
     * @return the distance to the nearest observation type
     */
    public double nearestDistance(ArrayList<Observation>[] pos) {
       	double smallestDist = Double.MAX_VALUE;
    	if (pos != null) {
    		for (int i = 0; i < pos.length; ++i) {
        		for (int r = 0; r < pos[i].size(); ++r) {
        			if (pos[i].get(r).sqDist < smallestDist) {
                		smallestDist = pos[i].get(r).sqDist;
                	}		
        		}
        	}	
    	}
    	return smallestDist;
    }
    
    public boolean canMove(String dir, StateObservation stateObs) {
    	int xOff = 0;
    	int yOff = 0;
    	switch (dir) {
	        case "left": xOff = -1; break;
	        case "right": xOff = 1; break;
	        case "up": yOff = -1; break;
	        case "down": yOff = 1; break;
    	}
                  
    	int blockSize = stateObs.getBlockSize();
    	ArrayList<Observation>[][] grid = stateObs.getObservationGrid();
    	int curX = (int)(stateObs.getAvatarPosition().x / blockSize);
    	int curY = (int)(stateObs.getAvatarPosition().y / blockSize);
    	curX += xOff;
    	curY += yOff;
    	//base case: can't move off the grid
    	if (curX >= grid.length || curY >= grid[curX].length) {
    		return false;
    	}
    	//check for immutable object in observation grid
    	for (int i = 0; i < grid[curX][curY].size(); ++i) {
    		//System.out.println(grid[curX][curY].get(i).category);
    		if (grid[curX][curY].get(i).category == 4) {//I believe 4 = immutable object
    			return false;
    		}
    	}
    	return true;
    }
    
    /**
     * sum the available resources
     * @param stateObs our state observation
     * @return the sum of our resources
     */
    public double getResourceSum(StateObservation stateObs) {
    	double sum = 0.0f;
    	for (double d : stateObs.getAvatarResources().values()) {
    	    sum += d;
    	}
    	return sum;
    }
        	
    /**
	 * check our inputs, propagate them, and return some output
	 * @return the array of keypresses comprising the action our agent wishes to perform
	 */
	@Override
	public ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
		//construct our input layer from each of our input conditions
    	double[] inputs = new double[] {
			stateObs.getAvatarPosition().x,
			stateObs.getAvatarPosition().y,
			stateObs.getAvatarSpeed(),
			getResourceSum(stateObs),
			nearestDistance(stateObs.getNPCPositions()),
			nearestDistance(stateObs.getMovablePositions()),
			nearestDistance(stateObs.getResourcesPositions()),
			nearestDistance(stateObs.getPortalsPositions()),
			nearestDistance(stateObs.getImmovablePositions()),
			nearestDistance(stateObs.getFromAvatarSpritesPositions()),
			canMove("left",stateObs) ? 1 : 0,
			canMove("right",stateObs) ? 1 : 0,
			canMove("up",stateObs) ? 1 : 0,
			canMove("down",stateObs) ? 1 : 0
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
        //System.out.println(Arrays.toString(inputs));
        //System.out.println(Arrays.toString(outputs));
        return act.get(largestOutput);
	}
}