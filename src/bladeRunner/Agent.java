package bladeRunner;

import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import agents.GameAgent;
import agents.hbfs.HBFSAgent;
import agents.mcts.MCTSAgent;
import agents.misc.AdjacencyMap;
import agents.misc.DrawingTools;
import agents.misc.GameClassifier;
import agents.misc.ITypeAttractivity;
import agents.misc.PersistentStorage;
import agents.misc.RewardMap;
import agents.misc.GameClassifier.GameType;
import agents.misc.pathplanning.PathPlanner;
import core.game.Observation;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import ontology.Types;
import tools.ElapsedCpuTimer;

/**
 * The main agent holding subagents.
 *
 */
public class Agent extends AbstractPlayer {

	/**
	 * The type of agent we intend to run.
	 */
	public enum AgentType {
		MCTS, BFS, MIXED
	}
	
	ArrayList<Observation>[][] grid;
	protected int blockSize; //only needed for the drawing
	
	public static final boolean isVerbose = true;

	/**
	 * The agent type we force the agent into.
	 */
	public AgentType forcedAgentType = AgentType.MIXED;

	/**
	 * Agents
	 */
	private static MCTSAgent mctsAgent = null;
	private static HBFSAgent hbfsAgent = null;
	private static GameAgent currentAgent = null;

	/**
	 * Agent switching properties
	 */
	private GameAgent previousAgent = null;
	private static int agentSwitchTicksRemaining = 0;

	/**
	 * Public constructor with state observation and time due.
	 * 
	 * @param so
	 *            state observation of the current game.
	 * @param elapsedTimer
	 *            Timer for the controller creation.
	 */
	public Agent(StateObservation so, ElapsedCpuTimer elapsedTimer) {
		// #################
		// PERSISTENT STORAGE
		// Get the actions in a static array.
		ArrayList<Types.ACTIONS> act = so.getAvailableActions();
		PersistentStorage.actions = new Types.ACTIONS[act.size()];
		for (int i = 0; i < PersistentStorage.actions.length; ++i) {
			PersistentStorage.actions[i] = act.get(i);
		}

		// initialize exploration reward map with 1
		PersistentStorage.rewMap = new RewardMap(so, 1);

		// initialize the adjacency map with the current state observation
		PersistentStorage.adjacencyMap = new AdjacencyMap(so);

		// initialize ItypeAttracivity object for starting situation
		PersistentStorage.iTypeAttractivity = new ITypeAttractivity(so);
		
		PersistentStorage.previousAvatarRessources = new HashMap<>();

		// Classify game
		GameClassifier.determineGameType(so);

		// use time that is left to build a tree or do BFS
		if ((GameClassifier.getGameType() == GameType.STOCHASTIC || forcedAgentType == AgentType.MCTS) && forcedAgentType != AgentType.BFS) {
			// Create the player.
			mctsAgent = new MCTSAgent(so, elapsedTimer, new Random());
			currentAgent = mctsAgent;
		} else {
			hbfsAgent = new HBFSAgent(so, elapsedTimer);
			currentAgent = hbfsAgent;
		}

	}

	/**
	 * Picks an action. This function is called every game step to request an
	 * action from the player.
	 * 
	 * @param stateObs
	 *            Observation of the current state.
	 * @param elapsedTimer
	 *            Timer when the action returned is due.
	 * @return An action for the current state
	 */
	public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
		Types.ACTIONS action = Types.ACTIONS.ACTION_NIL;
		
		//this is just for the drawing. comment it out, if you don't need it
		// DrawingTools.updateObservation(stateObs);
		
		try {
			action = currentAgent.act(stateObs, elapsedTimer);
		} catch (OutOfMemoryError e) {
			currentAgent.clearMemory();
			// action = currentAgent.act(stateObs, elapsedTimer);
		}

		// if agent is switched to another one
		if (agentSwitchTicksRemaining > 0 && previousAgent != null) {
			agentSwitchTicksRemaining--;
		} else {
			switchBack();
		}

		return action;
	}

	/**
	 * Switch to another agent for a limited number of ticks.
	 * 
	 * @note TODO: Not yet working, the agents need to be both initialized.
	 * 
	 * @param type
	 *            The type of agent we want to switch to.
	 */
	public void switchAgent(AgentType type) {
		switch (type) {
		case BFS:
			if (hbfsAgent != null) {
				currentAgent = hbfsAgent;
			} else {
				throw new NullPointerException("HBFS Agent needs to be initialized.");
			}
			break;
		case MCTS:
			if (mctsAgent != null) {
				currentAgent = mctsAgent;
			} else {
				throw new NullPointerException("MCTS Agent needs to be initialized.");
			}
			break;
		case MIXED:
		default:
			break;
		}
	}

	/**
	 * Switch to another agent for a limited number of ticks.
	 * 
	 * @note TODO: Not yet working, the agents need to be both initialized.
	 * 
	 * @param type
	 *            The type of agent we want to switch to.
	 * @param ticks
	 *            The number of ticks the agent should be switched.
	 */
	public static void switchAgentForTicks(AgentType type, int ticks) {
		agentSwitchTicksRemaining = ticks;
		switch (type) {
		case BFS:
			if (hbfsAgent != null) {
				currentAgent = hbfsAgent;
			} else {
				throw new NullPointerException("HBFS Agent needs to be initialized.");
			}
			break;
		case MCTS:
			if (mctsAgent != null) {
				currentAgent = mctsAgent;
			} else {
				throw new NullPointerException("MCTS Agent needs to be initialized.");
			}
			break;
		case MIXED:
		default:
			break;
		}
	}

	/**
	 * Switch the agent back to the previous agent.
	 * 
	 * @note TODO: Not yet working, the agents need to be both initialized.
	 */
	public void switchBack() {
		if (previousAgent != null) {
			currentAgent = previousAgent;
			previousAgent = null;
		}
	}
	
	
	
	
	
	/**
     * Gets the player the control to draw something on the screen.
     * It can be used for debug purposes.
     * The draw method of the agent is called by the framework (VGDLViewer) whenever it runs games visually
     * Comment this out, when you do not need it
     * We could draw anything!
     * @param g Graphics device to draw to.
     */
	/*
    public void draw(Graphics2D g)
    {
        DrawingTools.draw(g);
    }
    */ //if you want to use that, you also have to uncomment DrawingTools.updateObservation() in the act method
    
}
