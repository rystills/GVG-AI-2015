package ch.idsia.agents.learning;

import ch.idsia.agents.Agent;
import ch.idsia.benchmark.mario.environments.Environment;
import ch.idsia.evolution.Evolvable;
import ch.idsia.evolution.SmarterMLP;

public class SmarterMLPAgent implements Agent, Evolvable {

    public SmarterMLP mlp;
    private String name = "SmarterMLPAgent";
    //number of input nodes can be toggled; output nodes should remain at 6 (6 potential actions)
    final int numberOfOutputs = 6;
    final int numberOfInputs = 10;
    
    //standard integrated data
    private Environment environment;
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
    public SmarterMLPAgent() {
        mlp = new SmarterMLP(numberOfInputs, 10, numberOfOutputs);
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
    public void integrateObservation(Environment environment) {
        this.environment = environment;
        levelScene = environment.getLevelSceneObservationZ(zLevelScene);
        enemies = environment.getEnemiesObservationZ(zLevelEnemies);
        mergedObservation = environment.getMergedObservationZZ(1, 0);

        this.marioFloatPos = environment.getMarioFloatPos();
        this.enemiesFloatPos = environment.getEnemiesFloatPos();
        this.marioState = environment.getMarioState();

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
	 * check if there is a gap immediately in front of us
	 * @return: whether there is a gap immediately in front of us (true) or not (false)
	 */
    private boolean gapApproaching() {
	    if (getReceptiveFieldCellValue(marioCenter[0] + 1, marioCenter[1] + 1) == 0) {
	    	return true;
	    }
	    return false;
	}
	
	/**
	 * check if there is a wall 1 or 2 blocks in front of us
	 * @return: whether there is a wall in front of us (true) or not (false)
	 */
	private boolean wallApproaching() {
		if (getReceptiveFieldCellValue(marioCenter[0], marioCenter[1] + 1) != 0 || 
				getReceptiveFieldCellValue(marioCenter[0], marioCenter[1] + 2) != 0) {
	    	return true;
	    }
		return false;
	}
    
    /**
	 * check if there is an enemy 1 or 2 blocks in front of us
	 * @return: whether there is an enemy in front of us (true) or not (false)
	 */
	private boolean enemyApproaching() {
		for (int i = 0; i < enemiesFloatPos.length; i+=3) {
			float ex = enemiesFloatPos[i+1];
			//check if any enemies x vales are within a small area around mario
			if (ex <= 48 && ex >= -16) {
				return true;
			}
		}
		return false;
	}
	
	/**
     * check if we are approaching a gap that is still far from us
     * @return whether we are approaching a far gap (true) or not (false)
     */
    private boolean gapApproachingFar() {
    	return getReceptiveFieldCellValue(marioCenter[0] + 1, marioCenter[1] + 2) == 0;
    }
    
    /**
     * check if we are approaching a gap that is close to us
     * @return whether we are approaching a close gap (true) or not (false)
     */
    private boolean gapApproachingClose() {
    	return getReceptiveFieldCellValue(marioCenter[0] + 1, marioCenter[1] + 1) == 0;
    }
    
    /**
     * check if we are approaching a wall that is still far from us
     * @return whether we are approaching a far wall (true) or not (false)
     */
    private boolean wallApproachingFar() {
    	return getReceptiveFieldCellValue(marioCenter[0], marioCenter[1] + 2) != 0;
    }
    
    /**
     * check if we are approaching a wall that is close to us
     * @return whether we are approaching a close wall (true) or not (false)
     */
    private boolean wallApproachingClose() {
    	return getReceptiveFieldCellValue(marioCenter[0], marioCenter[1] + 1) != 0;
    }
    
    /**
	 * check if there is an enemy far from us
	 * @return: whether there is an enemy far in front of us (true) or not (false)
	 */
    private boolean enemyApproachingFar() {
    	for (int i = 0; i < enemiesFloatPos.length; i+=3) {
			float ex = enemiesFloatPos[i+1];
			//check if any enemies x vales are within a small area around mario
			if (ex <= 48 && ex >= 25) {
				return true;
			}
		}
		return false;
    }
    
    /**
	 * check if there is an enemy close to us
	 * @return: whether there is an enemy close in front of us (true) or not (false)
	 */
    private boolean enemyApproachingClose() {
    	for (int i = 0; i < enemiesFloatPos.length; i+=3) {
			float ex = enemiesFloatPos[i+1];
			//check if any enemies x vales are within a small area around mario
			if (ex <= 24 && ex >= -16) {
				return true;
			}
		}
		return false;
    }
	
	/**
	 * check if we moved down since last frame; if so, force trueJumpCounter to 0 and stop holding jump
	 * @return: whether or not we moved down since last frame and had trueJumpCounter > 1
	 */
	private boolean checkMovedDown() {
		//ignore trueJumpCounter <=1 so we can still wallkick (as during wallslide we are slowly moving down)
		if (marioFloatPos[1] > prevY) {
	    	return true;
		}
		return false;
	}
	
	/**
	 * check if we are about to stomp on an enemy; if so, hold jump
	 * @return: whether we are about to stomp on an enemy (true) or not (false)
	 */
	private boolean checkAboutToStomp() {
		//can't stomp an enemy while grouhnded
		if (isMarioOnGround) {
			return false;
		}
		//check if any enemies are within a tight bound of our x position (note that we ignore y position for simplicity)
		for (int i = 0; i < enemiesFloatPos.length; i+=3) {
			if (enemiesFloatPos[i+1] <= 16 && enemiesFloatPos[i+1] >= -8) {
		    	return true;
			}
		}
    	return false;
	}
	
	/**
	 * check whether or not mario is running
	 * @return: whether mario is running (true) or not (false)
	 */
	public boolean isMarioRunning() {
		//max speed when running appears to be a little over 10; 8 should be enough to get the NN on the right track
		return Math.abs(marioFloatPos[0] - prevX) >= 8; 
	}
    
	/**
	 * get the receptive field value at the specified coordinates
	 * @param x: the x coordinate in the field
	 * @param y: the y coordinate in the field
	 * @return the receptive field value at the specified coordinates
	 */
	public int getReceptiveFieldCellValue(int x, int y) {
	    if (x < 0 || x >= levelScene.length || y < 0 || y >= levelScene[0].length)
	        return 0;

	    return levelScene[x][y];
	}
	
	/**
	 * check our inputs, propagate them, and return some output
	 * @return the array of keypresses comprising the action our agent wishes to perform
	 */
    public boolean[] getAction() {
    	//update immediate non-standard persistent data
    	marioCenter = environment.getMarioReceptiveFieldCenter();
    	
    	//construct our input layer from each of our input conditions
    	double[] inputs = new double[] {
			gapApproaching() ? 1 : 0,
	    	wallApproaching() ? 1 : 0,
			enemyApproaching() ? 1 : 0,
			checkAboutToStomp() ? 1 : 0,
			checkMovedDown() ? 1 : 0,
			marioMode == 2 ? 1 : 0,
			marioMode == 0 ? 1 : 0,
			isMarioOnGround ? 1 : 0,
			isMarioAbleToJump ? 1 : 0,
			isMarioRunning() ? 0 : 1
    	};
    	
    	//construct our output layer by propagating our hidden layer from our inputs
        double[] outputs = mlp.propagate(inputs);
        
        //map outputs to 0/1 for keypresses
        boolean[] action = new boolean[numberOfOutputs];
        for (int i = 0; i < action.length; action[i] = outputs[i] > 0, ++i);
        
        //update late non-standard persistent data
	    prevY = marioFloatPos[1];
	    prevX = marioFloatPos[0];
	    
        return action;
    }
}