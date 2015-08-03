package benelot;

import core.game.Observation;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import ontology.Types;
import ontology.Types.ACTIONS;
import tools.ElapsedCpuTimer;

import java.util.ArrayList;
import java.util.Random;

import projects.HBFS.HBFSAgent;
import projects.MCTS.ITypeAttractivity;
import projects.MCTS.RewardMap;
import projects.MCTS.SingleMCTSPlayer;
import projects.MCTS.SingleTreeNode;
import tools.Vector2d;

/**
 * @Created with IntelliJ IDEA.
 * @User: ssamot
 * @Date: 14/11/13
 * @Time: 21:45
 * @detail This is a Java port from Tom Schaul's VGDL -
 *         https://github.com/schaul/py-vgdl
 */
public class Agent extends AbstractPlayer {

	// ## Parameters
	public static int ROLLOUT_DEPTH = 0;

	// running and fixed MCTS_DEPTH, first increments to counter the increment
	// of the depth of the cut trees. The later stays fixed
	public static int MCTS_DEPTH_FIX = 3;
	
	// fix the MCTS_DEPTH to the starting DEPTH
	public static int MCTS_DEPTH_RUN = MCTS_DEPTH_FIX;
	public static int MCTS_AVOID_DEATH_DEPTH = 2;
	public static double K = Math.sqrt(2);
	public static ACTIONS[] actions;

	public enum AgentType {
		MCTS, BFS, MIXED
	}

	public AgentType agentType = AgentType.MIXED;

	// an exploration reward map that is laid over the game-world to reward
	// places that haven't been visited lately
	public static RewardMap rewMap;

	// HashMap of iTypeAttractivity for start situation
	// TODO: Maybe create List of AttractivityMaps for different game situations
	// (e.g. avatar has found sword/has eaten mushroom/has a lot of honey)
	public static ITypeAttractivity iTypeAttractivity;

	// keeps track of the reward at the start of the MCTS search
	public static double startingReward = 0;
	public static double numberOfBlockedMovables = 0;
	
	public boolean isStochastic = false;
	
	public int oldAction = -2;

	/**
	 * Random generator for the agent.
	 */
	private SingleMCTSPlayer mctsPlayer;
	private HBFSAgent bfAgent;

	/**
	 * Public constructor with state observation and time due.
	 * 
	 * @param so
	 *            state observation of the current game.
	 * @param elapsedTimer
	 *            Timer for the controller creation.
	 */
	public Agent(StateObservation so, ElapsedCpuTimer elapsedTimer) {
		// Get the actions in a static array.
		ArrayList<Types.ACTIONS> act = so.getAvailableActions();
		actions = new Types.ACTIONS[act.size()];
		for (int i = 0; i < actions.length; ++i) {
			actions[i] = act.get(i);
		}

		// Create the player.
		mctsPlayer = new SingleMCTSPlayer(new Random());

		// initialize exploration reward map with 1
		rewMap = new RewardMap(so, 1);

		// initialize ItypeAttracivity object for starting situation
		iTypeAttractivity = new ITypeAttractivity(so);

		// Advance a bit to check if stochastic
		isStochastic = false;
		StateObservation testState1 = so.copy();
		StateObservation testState2 = so.copy();
		for (int ii = 1; ii < 4; ii++) {
			testState1.advance(Types.ACTIONS.ACTION_NIL);
			testState2.advance(Types.ACTIONS.ACTION_NIL);
			
			if(!testState1.equiv(testState2)){
				isStochastic = true;
				break;
			}
		}
		
		if (isStochastic) {
			System.out.println("Game seems to be stochastic");
		}
		else{
			MCTS_DEPTH_RUN += 20;
			System.out.println("Game seems to be deterministic");
		}


		// use time that is left to build a tree or do BFS
		if ((isStochastic || agentType == AgentType.MCTS)
				&& agentType != AgentType.BFS) {
			mctsPlayer.init(so);
			mctsPlayer.run(elapsedTimer);
		} else {
			bfAgent = new HBFSAgent(so, elapsedTimer);
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
	public Types.ACTIONS act(StateObservation stateObs,
			ElapsedCpuTimer elapsedTimer) {

		if (agentType == AgentType.BFS
				|| (agentType == AgentType.MIXED && !isStochastic)) {
			Types.ACTIONS bfaction = this.bfAgent.act(stateObs, elapsedTimer);
			return bfaction;
		} else {

			// Heuristic: change the reward in the exploration reward map of the
			// visited current position
			Vector2d avatarPos = stateObs.getAvatarPosition();

			// increment reward at all unvisited positions and decrement at
			// current position
			rewMap.incrementAll(0.001);
			rewMap.setRewardAtWorldPosition(avatarPos, -0.3);

			// rewMap.print();

			// Heuristic: Punish the exploration area where enemies have been
			// and reward it if the enemies are attractive
			// TODO: This could also be done for resources and one could reward
			// the surroundings (with a reward gradient, see IDEA below)
			// TODO: Or maybe it would be better not to reward the positions
			// where the npcs were, but always the position and surroundings,
			// where they currently are
			// which would require a lot of time (I tried it for all Sprites,
			// but it was too inefficient)
			// Heuristic (IDEA): Perhaps increase reward towards positions that
			// have
			// a resource. -> some sort of diffusion model with the resources as
			// positive sources and the enemies as negative sources to create a
			// reward
			// gradient.

			boolean rewardNPCs = true;
			if (rewardNPCs) {
				ArrayList<Observation>[] npcPositions = null;
				npcPositions = stateObs.getNPCPositions();
				double npcAttractionValue = 0;
				if (npcPositions != null) {
					for (ArrayList<Observation> npcs : npcPositions) {
						if (npcs.size() > 0) {
							Vector2d npcPos = npcs.get(0).position;
							try {
								npcAttractionValue = iTypeAttractivity.get(npcs
										.get(0).itype);
							} catch (java.lang.NullPointerException e) {
								iTypeAttractivity
										.putNewUniqueItype(npcs.get(0));
								npcAttractionValue = iTypeAttractivity.get(npcs
										.get(0).itype);
							}
							if (Math.abs(rewMap
									.getRewardAtWorldPosition(npcPos)) < 1) {
								rewMap.incrementRewardAtWorldPosition(npcPos,
										npcAttractionValue * 0.02);
							}
						}
					}
				}
			}

			boolean rewardRecources = true;
			if (rewardRecources) {
				ArrayList<Observation>[] resPositions = null;
				resPositions = stateObs.getNPCPositions();
				double resAttractionValue = 0;
				if (resPositions != null) {
					for (ArrayList<Observation> ress : resPositions) {
						if (ress.size() > 0) {
							Vector2d resPos = ress.get(0).position;
							try {
								resAttractionValue = iTypeAttractivity.get(ress
										.get(0).itype);
							} catch (java.lang.NullPointerException e) {
								iTypeAttractivity
										.putNewUniqueItype(ress.get(0));
								resAttractionValue = iTypeAttractivity.get(ress
										.get(0).itype);
							}
							if (Math.abs(rewMap
									.getRewardAtWorldPosition(resPos)) < 1) {
								rewMap.incrementRewardAtWorldPosition(resPos,
										resAttractionValue * 0.02);
							}
						}
					}
				}
			}

			int useOldTree = 1;
			if (useOldTree == 1) {
				// Sets a new tree with the children[oldAction] as the root
				mctsPlayer.initWithOldTree(stateObs, oldAction);
			} else {
				// Set the state observation object as the new root of the tree.
				// (Forgets the whole tree)
				mctsPlayer.init(stateObs);
			}

			startingReward = stateObs.getGameScore();

			numberOfBlockedMovables = SingleTreeNode.trapHeuristic(stateObs);

			// Determine the action using MCTS...
			int action = mctsPlayer.run(elapsedTimer);
			// if (stateObs.getGameTick() % 2 == 0) {
			// action = -2;
			// }
			if (action > -2) {
				MCTS_DEPTH_RUN += useOldTree;
			}

			// there is a problem when the tree is so small that the chosen
			// children don't have any grand children,
			// in this case Danny's isDeadEnd method will give back a true based
			// on the "fear_unknown" input. Therefore,
			// I treat this waiting as a thinking step, where we expand the old
			// tree instead of creating a complete
			// new tree that also leads to the same problem -> the guy is stuck.

			if (action == -1)
				action = -2;

			oldAction = action;

			// ... and return it.
			if (action == -2 || action == -1) {
				return Types.ACTIONS.ACTION_NIL;
			} else {
				return actions[action];
			}

		} // end if isstochastic

	}
}