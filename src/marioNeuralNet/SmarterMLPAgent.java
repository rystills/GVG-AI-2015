package marioNeuralNet;

import java.util.ArrayList;

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
     * find the distance to the nearest NPC
     * @param stateObs our state observation
     * @return the distance to the nearest npc
     */
    public double nearestNPCDistance(StateObservation stateObs) {
    	double smallestDist = Double.MAX_VALUE;
    	ArrayList<Observation>[] pos = stateObs.getNPCPositions();
    	for (int i = 0; i < pos.length; ++i) {
    		for (int r = 0; r < pos[i].size(); ++r) {
    			if (pos[i].get(r).sqDist < smallestDist) {
            		smallestDist = pos[i].get(r).sqDist;
            	}		
    		}
    	}
    	return smallestDist;
    }
    
    /**
     * find the distance to the nearest movable
     * @param stateObs our state observation
     * @return the distance to the nearest npc
     */
    public double nearestMovableDistance(StateObservation stateObs) {
    	double smallestDist = Double.MAX_VALUE;
    	ArrayList<Observation>[] pos = stateObs.getMovablePositions();
    	for (int i = 0; i < pos.length; ++i) {
    		for (int r = 0; r < pos[i].size(); ++r) {
    			if (pos[i].get(r).sqDist < smallestDist) {
            		smallestDist = pos[i].get(r).sqDist;
            	}		
    		}
    	}
    	return smallestDist;
    }
    
    /**
     * find the distance to the nearest resource
     * @param stateObs our state observation
     * @return the distance to the nearest npc
     */
    public double nearestResourceDistance(StateObservation stateObs) {
    	double smallestDist = Double.MAX_VALUE;
    	ArrayList<Observation>[] pos = stateObs.getResourcesPositions();
    	for (int i = 0; i < pos.length; ++i) {
    		for (int r = 0; r < pos[i].size(); ++r) {
    			if (pos[i].get(r).sqDist < smallestDist) {
            		smallestDist = pos[i].get(r).sqDist;
            	}		
    		}
    	}
    	return smallestDist;
    }
    
    /**
     * find the distance to the nearest portal
     * @param stateObs our state observation
     * @return the distance to the nearest npc
     */
    public double nearestPortalDistance(StateObservation stateObs) {
    	double smallestDist = Double.MAX_VALUE;
    	ArrayList<Observation>[] pos = stateObs.getPortalsPositions();
    	for (int i = 0; i < pos.length; ++i) {
    		for (int r = 0; r < pos[i].size(); ++r) {
    			if (pos[i].get(r).sqDist < smallestDist) {
            		smallestDist = pos[i].get(r).sqDist;
            	}		
    		}
    	}
    	return smallestDist;
    }
    
    /**
     * find the distance to the nearest immovable
     * @param stateObs our state observation
     * @return the distance to the nearest npc
     */
    public double nearestImmovableDistance(StateObservation stateObs) {
    	double smallestDist = Double.MAX_VALUE;
    	ArrayList<Observation>[] pos = stateObs.getImmovablePositions();
    	for (int i = 0; i < pos.length; ++i) {
    		for (int r = 0; r < pos[i].size(); ++r) {
    			if (pos[i].get(r).sqDist < smallestDist) {
            		smallestDist = pos[i].get(r).sqDist;
            	}		
    		}
    	}
    	return smallestDist;
    }
    
    /**
     * find the distance to the nearest sprite
     * @param stateObs our state observation
     * @return the distance to the nearest npc
     */
    public double nearestSpriteDistance(StateObservation stateObs) {
    	double smallestDist = Double.MAX_VALUE;
    	ArrayList<Observation>[] pos = stateObs.getFromAvatarSpritesPositions();
    	for (int i = 0; i < pos.length; ++i) {
    		for (int r = 0; r < pos[i].size(); ++r) {
    			if (pos[i].get(r).sqDist < smallestDist) {
            		smallestDist = pos[i].get(r).sqDist;
            	}		
    		}
    	}
    	return smallestDist;
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
			stateObs.getAvatarResources().values().stream().mapToDouble(u -> u).sum(), //not sure about this calculation
			nearestNPCDistance(stateObs),
			nearestMovableDistance(stateObs),
			nearestResourceDistance(stateObs),
			nearestPortalDistance(stateObs),
			nearestImmovableDistance(stateObs),
			nearestSpriteDistance(stateObs),
			stateObs.getAvatarSpeed()
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