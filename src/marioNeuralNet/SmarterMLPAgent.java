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
    
    //standard integrated data
    //private Environment environment;
    protected byte[][] levelScene;
    protected byte[][] enemies;
    protected byte[][] mergedObservation;
    protected float[] marioFloatPos = null;
    protected float[] enemiesFloatPos = null;
    protected int[] marioState = null;
    protected int marioStatus;
    protected int marioMode;
    protected boolean isMarioOnGround;
    protected boolean isMarioAbleToJump;
    protected boolean isMarioAbleToShoot;
    protected boolean isMarioCarrying;
    protected int getKillsTotal;
    protected int getKillsByFire;
    protected int getKillsByStomp;
    protected int getKillsByShell;

    //adjust for varying environment information granularity (see: marioai.org/marioaibenchmark/zLevels)
    int zLevelScene = 1;
    int zLevelEnemies = 0;

	//non-standard persistent variables for integrated stateful logic
    float prevX = 0; //x value last frame
    float prevY = 0; //y value last frame
    protected int[] marioCenter; //receptive field center

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
    public void integrateObservation() {
//        this.environment = environment;
//        levelScene = environment.getLevelSceneObservationZ(zLevelScene);
//        enemies = environment.getEnemiesObservationZ(zLevelEnemies);
//        mergedObservation = environment.getMergedObservationZZ(1, 0);
//
//        this.marioFloatPos = environment.getMarioFloatPos();
//        this.enemiesFloatPos = environment.getEnemiesFloatPos();
//        this.marioState = environment.getMarioState();

        //many of these go unused, but are left in for convenient potential future use
        marioStatus = marioState[0];
        marioMode = marioState[1];
        isMarioOnGround = marioState[2] == 1;
        isMarioAbleToJump = marioState[3] == 1;
        isMarioAbleToShoot = marioState[4] == 1;
        isMarioCarrying = marioState[5] == 1;
        getKillsTotal = marioState[6];
        getKillsByFire = marioState[7];
        getKillsByStomp = marioState[8];
        getKillsByShell = marioState[9];
    }

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