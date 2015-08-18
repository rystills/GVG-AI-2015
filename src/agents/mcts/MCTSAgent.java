package agents.mcts;

import agents.GameAgent;
import agents.misc.ObservationTools;
import agents.misc.PersistentStorage;
import core.game.Observation;
import core.game.StateObservation;
import tools.ElapsedCpuTimer;
import tools.Vector2d;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import ontology.Types;
import ontology.Types.ACTIONS;
import agents.misc.pathplanning.PathPlanner;

public class MCTSAgent extends GameAgent {
	/**
	 * Root of the tree.
	 */
	public MCTSNode m_root;

	/**
	 * Random generator.
	 */
	public Random m_rnd;
	
	public static Vector2d startingPos;

	public int nodeQty;
	

	// additional actions names
	public final int ADD_NEW_ROOT_NODE = -1;
	public final int KEEP_COMPLETE_OLD_TREE = -2;

	
	public int oldAction = KEEP_COMPLETE_OLD_TREE;

	public static HashMap<Integer, PathPlanner> pathPlannerMaps;
	
	public static int treePolMode; // 0: old mode with MCTS, 1: goaldriven mode
	public static int goalItype;
	public static int idOfGoal;
	public static int GoalDrivenTreePolCounter;
	public static int GoalDrivenTreePolCounter_Max = 10;
	public static double chanceOfGoalDrivenPlanning = 0.20;

	/**
	 * Creates the MCTS player with a sampleRandom generator object.
	 * 
	 * @param a_rnd
	 *            sampleRandom generator object.
	 */
	public MCTSAgent(StateObservation so, ElapsedCpuTimer elapsedTimer,
			Random a_rnd) {
		m_rnd = a_rnd;
		m_root = new MCTSNode(a_rnd);
		init(so);
		run(elapsedTimer);
		nodeQty = 0;
		treePolMode = 0;
		idOfGoal = 0;
		goalItype = 0;
		GoalDrivenTreePolCounter = 0;
	}

	/**
	 * Inits the tree with the new observation state in the root.
	 * 
	 * @param a_gameState
	 *            current state of the game.
	 */
	public void initNew(StateObservation a_gameState) {
		// Set the game observation to a newly root node.
		m_root = new MCTSNode(m_rnd);
		m_root.state = a_gameState;
		startingPos = a_gameState.getAvatarPosition();

	}

	public void init(StateObservation a_gameState) {
		// Set the game observation to a newly root node.
		m_root = new MCTSNode(m_rnd);
		m_root.state = a_gameState;
		startingPos = a_gameState.getAvatarPosition();

	}

	public void initWithOldTree(StateObservation a_gameState, int action) {
		/*
		 * Here we create a new root-tree for the next search query based on the
		 * old tree. The old tree gets cut at the chosen action. The depth of
		 * that tree is not changed such that it grows throughout the game.
		 * Therefore, the maximal MCTS_Depth also grows
		 */
		startingPos = a_gameState.getAvatarPosition();
		if (action == KEEP_COMPLETE_OLD_TREE) {
			//int oldDepth = m_root.m_depth;
			// This means we want to keep the old tree and just update the state
			m_root.state = a_gameState;
			//m_root.m_depth = oldDepth;
		} else {
			if (action == ADD_NEW_ROOT_NODE) {
				m_root = new MCTSNode(m_rnd);
				m_root.state = a_gameState;
				
			} else {
				// cut old tree
				m_root = m_root.children[action];
				m_root.state = a_gameState;
				m_root.parent = null;
				// m_root.startingRew = a_gameState.getGameScore();

				// adapting the tree depth is not really feasible within the
				// time constraints
				// m_root.correctDepth();
			}
		}
	}

	/**
	 * Runs MCTS to decide the action to take. It does not reset the tree.
	 * 
	 * @param elapsedTimer
	 *            Timer when the action returned is due.
	 * @return the action to execute in the game.
	 */
	public int run(ElapsedCpuTimer elapsedTimer) {
		
		// create the different PathPlanningMaps for the different itypes that we see at the moment.
		initPathPlannerMaps(m_root.state);
		
		PersistentStorage.previousAvatarRessources = m_root.state.getAvatarResources();
		
		
		// set the counter for the goalDriven way of planning
		if(GoalDrivenTreePolCounter > 0){
			GoalDrivenTreePolCounter--;
			if(GoalDrivenTreePolCounter == 0){
				treePolMode = 0;
				idOfGoal = 0;
				goalItype = 0;
			}
		}
		//decide on a planning strategy
		if(m_rnd.nextDouble() < chanceOfGoalDrivenPlanning && treePolMode == 0){
			treePolMode = 1;
			GoalDrivenTreePolCounter =  GoalDrivenTreePolCounter_Max;
			chooseANewGoal(m_root.state);
		}
	System.out.println(GoalDrivenTreePolCounter + "  "+ treePolMode + "   "+ idOfGoal+ "  "+   goalItype);
		
		// Do the search within the available time.
		m_root.mctsSearch(elapsedTimer);

		// Determine the best action to take and return it.
		// int action = m_root.mostVisitedAction();
		
		int action = m_root.bestAction();
//			for (int i = 1; i<= m_root.children.length ; i++ ){
//				if(m_root.children[i-1] != null){
//					System.out.print("  val"+i+": "+ m_root.children[i-1].totValue);
//				}
//			}
//
//			System.out.println();
//		 System.out.println(" RewOfSys :" + m_root.value(m_root.state));
//		 System.out.println(" action :" + action);
		// if (action >= 0){
		// System.out.println(" isdeath? :" +
		// m_root.children[action].isLoseState());
		// System.out.println(" isdeadend? :" +
		// m_root.children[action].isDeadEnd(2, true));
		// }
//		// if (action >= 0)
//		 System.out.println(" mroot :" + m_root.nVisits +" visits and the choosen child has "+ m_root.children[action].nVisits);

		// System.out.println("Number of nodes: " + m_root.countNodes());

		return action;
	}
	
	

	@Override
	public ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
		// Heuristic: change the reward in the exploration reward map of the
		// visited current position
		Vector2d avatarPos = stateObs.getAvatarPosition();
		startingPos = avatarPos;
		
		// increment reward at all unvisited positions and decrement at
		// current position
		PersistentStorage.rewMap.incrementAll(0.001);
		PersistentStorage.rewMap.setRewardAtWorldPosition(avatarPos, -0.3);
		PersistentStorage.rewMap.decrementAtPos(avatarPos, -0.01);

		// rewMap.print();

		/*
		 * Heuristic: Punish the exploration area where enemies have been // and
		 * reward it if the enemies are attractive
		 * 
		 * @note TODO: This could also be done for resources and one could
		 * reward the surroundings (with a reward gradient, see IDEA below)
		 * 
		 * @note TODO: Or maybe it would be better not to reward the positions
		 * where the npcs were, but always the position and surroundings, where
		 * they currently are which would require a lot of time (I tried it for
		 * all Sprites, but it was too inefficient)
		 * 
		 * @note Heuristic (IDEA): Perhaps increase reward towards positions
		 * that have a resource. -> some sort of diffusion model with the
		 * resources as positive sources and the enemies as negative sources to
		 * create a reward gradient.
		 */

		boolean rewardNPCs = false;
		if (rewardNPCs) {
			ArrayList<Observation>[] npcPositions = null;
			npcPositions = stateObs.getNPCPositions();
			double npcAttractionValue = 0;
			if (npcPositions != null) {
				for (ArrayList<Observation> npcs : npcPositions) {
					for (int i = 0; i < npcs.size(); i++) {
						Vector2d npcPos = npcs.get(i).position;

						npcAttractionValue = PersistentStorage.iTypeAttractivity
								.putIfAbsent(npcs.get(i));

						if (Math.abs(PersistentStorage.rewMap
								.getRewardAtWorldPosition(npcPos)) < 1) {
							PersistentStorage.rewMap
									.incrementRewardAtWorldPosition(npcPos,
											npcAttractionValue * 0.02);
						}
					}
				}
			}
		}

		boolean rewardResources = true;
		if (rewardResources) {
			ArrayList<Observation>[] resPositions = null;
			resPositions = stateObs.getNPCPositions();
			double resAttractionValue = 0;
			if (resPositions != null) {
				for (ArrayList<Observation> ress : resPositions) {
					for (int i = 0; i < ress.size(); i++) {
						Vector2d resPos = ress.get(i).position;

						resAttractionValue = PersistentStorage.iTypeAttractivity
								.putIfAbsent(ress.get(i));

						if (Math.abs(PersistentStorage.rewMap
								.getRewardAtWorldPosition(resPos)) < 1) {
							PersistentStorage.rewMap
									.incrementRewardAtWorldPosition(resPos,
											resAttractionValue * 0.02);
						}
					}
				}
			}
		}

		boolean useOldTree = true;
		if (useOldTree) {
			// Sets a new tree with the children[oldAction] as the root
			initWithOldTree(stateObs, oldAction);
		} else {
			// Set the state observation object as the new root of the tree.
			// (Forgets the whole tree)
			init(stateObs);
		}

		PersistentStorage.startingReward = stateObs.getGameScore();

		PersistentStorage.numberOfBlockedMovables = MCTSNode
				.trapHeuristic(stateObs);

		// Determine the action using MCTS...
		int action = run(elapsedTimer);
		// if (stateObs.getGameTick() % 2 == 0) {
		// action = KEEP_COMPLETE_TREE;
		// }
		if (action > ADD_NEW_ROOT_NODE && useOldTree) {
			PersistentStorage.MCTS_DEPTH_RUN += 1;
		}

		/*
		 * there is a problem when the tree is so small that the chosen children
		 * don't have any grand children, in this case Danny's isDeadEnd method
		 * will give back a true based on the "fear_unknown" input. Therefore, I
		 * treat this waiting as a thinking step, where we expand the old tree
		 * instead of creating a complete new tree that also leads to the same
		 * problem -> the guy is stuck.
		 */

		if (action == ADD_NEW_ROOT_NODE)
			action = KEEP_COMPLETE_OLD_TREE;
		
		
		oldAction = action;
		
		PersistentStorage.lastGameState = stateObs;
		if(action >0)
			PersistentStorage.lastWinLoseExpectation =m_root.children[action].totValue;
		else
			PersistentStorage.lastWinLoseExpectation = -1;

		
		// ... and return it.
		if (action == KEEP_COMPLETE_OLD_TREE || action == ADD_NEW_ROOT_NODE) {
			return Types.ACTIONS.ACTION_NIL;
		} else {
			return PersistentStorage.actions[action];
		}

	}
	
	
	public void initPathPlannerMaps(StateObservation state){

		pathPlannerMaps = new HashMap<Integer,PathPlanner>();
		// add Npc maps
		updatePPMaps(1,state);
		// add Ressource maps
		updatePPMaps(2,state);
		// add movable maps
		updatePPMaps(3,state);
		
		
	}
	
	public void updatePPMaps(int movNpcOrRes, StateObservation state){
		
		Vector2d posAvatar = state.getAvatarPosition();
		int blockSize = state.getBlockSize();
		int avaX = floorDiv((int) (posAvatar.x + 0.1), blockSize);
		int avaY = floorDiv((int) (posAvatar.y + 0.1), blockSize);
		
		ArrayList<Observation>[] movPos = null;
		
		// decide between NPC, Ressource and Movables
		if(movNpcOrRes == 1){
			movPos = state.getNPCPositions(posAvatar);
		}
		else{
			if(movNpcOrRes == 2){
				movPos = state.getResourcesPositions(posAvatar);
			}
			else{
				if(movNpcOrRes == 3){
					movPos = state.getMovablePositions(posAvatar);
				}
			}
		}
		
		if (movPos != null) {
			for (ArrayList<Observation> mov : movPos) {
				if (mov.size() > 0) {
					//for(int i = 0; i< res.size(); i++){
					//only look at the closest rewarding/punishing npc
					for(int i = 0; i< 1; i++){

						double movAttractionValue = 0;
						try {
							movAttractionValue = PersistentStorage.iTypeAttractivity
									.get(mov.get(i).itype);
						} catch (java.lang.NullPointerException e) {
							PersistentStorage.iTypeAttractivity
							.putIfAbsent(mov.get(i));
							movAttractionValue = PersistentStorage.iTypeAttractivity
									.get(mov.get(i).itype);
						}
						movAttractionValue = PersistentStorage.iTypeAttractivity
								.get(mov.get(i).itype);
						// update the pathplannerMaps for the closest movables
						PathPlanner pp = new PathPlanner();
						Vector2d movPosition = mov.get(i).position;
						int movX = floorDiv((int) (movPosition.x + 0.1), blockSize);
						int movY = floorDiv((int) (movPosition.y + 0.1), blockSize);
						pp.updateStart(avaX,avaY);
						pp.updateGoal(movX, movY);
						pp.updateWays();
						pathPlannerMaps.put(mov.get(i).itype, pp);
					}
				}
			}
		}

	}
	
	public static void chooseANewGoal(StateObservation state){
//		int newGoalID = 0;
//		int newGoalItype = 0;
//		
		
		// TODO: figgure out a way to make a list of all the itypes we could visit (excluding walls, and empty space)
		// then set some attractivity for them based on distance / itypeAttractivity
		// form a probability from that attractivity
		// take one sample from that probability
		
		HashMap<Integer, Integer> ObsList = ObservationTools.getObsList(state);
		// choose goal from the list of itypes. persist for n number of ticks
		chooseGoalFromObsList(ObsList,state);
		
		
//		idOfGoal = newGoalID;
//		goalItype =newGoalItype;
		
	}
	
	
	public static void chooseGoalFromObsList(HashMap<Integer,Integer> ObsList, StateObservation currentState){
		int goal = -1;
		Vector2d pos = currentState.getAvatarPosition();
		// to choose the object (goal) to pursue
		// get up to 10 closest objects based on Euclidean distance
		// get closest 3 npc s
		
		// get most attractive npc		
		ArrayList<Observation>[] npcPositions = null;
		npcPositions = currentState.getNPCPositions(pos);
		int attractiveNPCitype = getMostAttractiveItype(npcPositions);
		int attractiveNPCid = getIDofItpye(npcPositions,attractiveNPCitype);
		
		// get most attractive resource
		ArrayList<Observation>[] resPositions = null;
		resPositions = currentState.getResourcesPositions(pos);
		int attractiveResitype = getMostAttractiveItype(resPositions);
		int attractiveResid = getIDofItpye(resPositions,attractiveResitype);
		
		// get most attractive movable
		ArrayList<Observation>[] movablePositions = null;
		movablePositions = currentState.getMovablePositions(pos);
		int attractiveMovableitype = getMostAttractiveItype(movablePositions);
		int attractiveMovableid = getIDofItpye(movablePositions,attractiveMovableitype);
		
		// go towards the one with the highest attractivity out of the above (=goal)
		// TODO pick one in random. May be, first take 2 of each type and pick one of 6 in random?
		int randomInt = (int )(Math.random() * 3);
		
		switch (randomInt) {
			case 0: 	idOfGoal = attractiveMovableid;
						goalItype = attractiveMovableitype;
			
			case 1: 	idOfGoal = attractiveNPCid;
						goalItype = attractiveNPCitype;
			
			default:	idOfGoal = attractiveResid;
						goalItype = attractiveResitype;	
		}
		
		System.out.println("goalItype  :" + goalItype + "   goalID"+ idOfGoal);
//		return goal;
	}
	
	public static int getMostAttractiveItype(ArrayList<Observation>[] obsPositions){
		// return the id of the most attractive object in the given list of observations
		int attractiveID = -1;
		if (obsPositions==null){
			return attractiveID;
		}
		double attractiveItype_val = - Double.MAX_VALUE;
		int attractiveItype = -1;
		// choose a random itype, Thus we dont chooe more often things with lots of different instances. 
		for (ArrayList<Observation> npcs : obsPositions) {
			if (npcs.size() > 0) {
				
				double npcAttractionValue = 0;
				try {
					npcAttractionValue = PersistentStorage.iTypeAttractivity
							.get(npcs.get(0).itype);
				} catch (java.lang.NullPointerException e) {
					PersistentStorage.iTypeAttractivity
					.putIfAbsent(npcs.get(0));
					npcAttractionValue = PersistentStorage.iTypeAttractivity
							.get(npcs.get(0).itype);
				}
				
				if(npcAttractionValue == -2){
					// dont try to go to the killing npc
					
				}
				else{
					double interestFact = Math.random();
					if(interestFact > attractiveItype_val){
						attractiveItype = npcs.get(0).itype;
						attractiveItype_val = interestFact;
					}
				}
					
//				//				 for(int i = 0; i< npcs.size(); i++){
//				// look at the most rewarding/punishing npc
//				for(int i = 0; i<npcs.size(); i++){
//
//					double npcAttractionValue = 0;
//					try {
//						npcAttractionValue = PersistentStorage.iTypeAttractivity
//								.get(npcs.get(i).itype);
//					} catch (java.lang.NullPointerException e) {
//						PersistentStorage.iTypeAttractivity
//						.putIfAbsent(npcs.get(i));
//						npcAttractionValue = PersistentStorage.iTypeAttractivity
//								.get(npcs.get(i).itype);
//					}
//					
//					// update best attration value
//					if(npcAttractionValue > bestNpcAttractionValue){
//						bestNpcAttractionValue = npcAttractionValue;
//						attractiveID = i;
//					}
						
//				}
			}
		}
		
		return attractiveItype;
				
//		// get the ID
//		if(attractiveID  > 0){
//			for (ArrayList<Observation> npcs : obsPositions) {
//				if (npcs.size() > 0) {
//					if(npcs.get(0).itype == attractiveID){
//						int randID =  (int )(Math.random() * npcs.size() );
//						attractiveID = npcs.get(randID).obsID;
//					}
//						
//					
//				}
//				}
//			
//			
		//		}
		//		
		//		return attractiveID;
	}

	public static int getIDofItpye(ArrayList<Observation>[] obsPositions , int attractiveItype){
		// get the ID
		int attractiveID = -1;
		if(attractiveItype  > 0){
			for (ArrayList<Observation> obs : obsPositions) {
				if (obs.size() > 0) {
					if(obs.get(0).itype == attractiveItype){
						int randID =  (int )(Math.random() * obs.size() );
						attractiveID = obs.get(randID).obsID;
					}
				}
			}
		}
		return attractiveID;
	}
	
	
	public static int floorDiv(int x, int y) {
		int r = x / y;
		// if the signs are different and modulo not zero, round down
		if ((x ^ y) < 0 && (r * y != x)) {
			r--;
		}
		return r;
	}


	public void clearMemory() {
		// TODO Implement what to do when we run out of memory.

	}

}