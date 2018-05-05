package marioNeuralNet;

import java.util.Random;

import core.ArcadeMachine;
import misc.runners.GameLevelPair;
import misc.runners.RunConfig;

/*
 * Possible Games:
 * ALIENS, BOULDERDASH, BUTTERFLIES, CHASE, FROGS, MISSILECOMMAND,
 * PORTALS, SOKOBAN, SURVIVEZOMBIES, ZELDA
 */

public class marioNeuralNetRunner {

	public static void main(String[] args) throws Exception {		
		// train 10 generations
		trainGenerations(12);
		// play game visually once
//		config.setRepetitions(1);
//		runGamesVisually(config);
	}
	
	//#outputs for FROGS: 4
    static Evolvable initial = new SmarterMLPAgent(4);
    static SmarterES es = new SmarterES(initial, 50, 25); //50 total population, with 25 parents = 25 children (even split)
	
	/**
	 * train the NN for the specified #generations, then do a visual run with the results
	 * @param genNum the number of generations to train
	 */
	public static void trainGenerations(int genNum) throws Exception {
		//~CONFIG~
		RunConfig config = new RunConfig();
		//config.addGameLevel("qlearnMaze", 0);
		config.addGameLevel(RunConfig.GamesTraining2014.FROGS, 1);
		//config.addGameLevel("qlearnMaze",0);//RunConfig.GamesTraining2014.qlearnMaze, 1);
		
		config.setController(NNProxyAgent.class.getCanonicalName());
		config.setSaveActions(true);
		
		//~TRAIN~
        float mutationMagnitude = .3f; //starting mutation magnitude, if using scaling mutation
		for (int i = 0; i < genNum; ++i) {
			es.nextGeneration(mutationMagnitude);
			//evaluate all members of population
			for (int r = 0; r < es.population.length; ++r) {
				System.out.print(String.format("Evaluating generation %d/%d pop member %d/%d",i+1,genNum,r+1,es.population.length));
				NNProxyAgent.curAgent = (SmarterMLPAgent)es.population[r];
				es.population[r].reset();
				double score = runOneGame(config);
	            es.fitness[r] = (float)score;
			}
			es.shuffle();
			es.sortPopulationByFitness();
		}
		
		//~RESULT~
		NNProxyAgent.curAgent = (SmarterMLPAgent)es.population[0];
		runGamesVisually(config);
	}

	/**
	 * Run the configured games with the configured controller and show the game
	 * visually.
	 * 
	 * @param config
	 *            The run configuration containing the game details.
	 */
	public static void runGamesVisually(RunConfig config) {
		for (GameLevelPair<String, String[]> gameLevelPair : config
				.getGameLevels()) {
			for (String level : gameLevelPair.level) {
				for (int repetition = 0; repetition < config.getRepetitions(); repetition++) {
					String actionsFile = "actions_game_" + gameLevelPair.game
							+ "_lvl_" + level + "_r" + repetition + "_"
							+ RunConfig.getTimestampNow() + ".txt";
					ArcadeMachine.runOneGame(RunConfig
							.getGamePath(gameLevelPair.game), RunConfig
							.getGameLevelPath(gameLevelPair.game, level), true,
							config.getController(),
							(config.isSaveActions()) ? actionsFile : null,
							new Random().nextInt());
				}
			}
		}

	}
	
	/**
	 * run a single game and get the score
	 * @param config the game config to use
	 */
	public static double runOneGame(RunConfig config) {
		for (GameLevelPair<String, String[]> gameLevelPair : config
				.getGameLevels()) {
			for (String level : gameLevelPair.level) {
				for (int repetition = 0; repetition < config.getRepetitions(); repetition++) {
					String actionsFile = "actions_game_" + gameLevelPair.game
							+ "_lvl_" + level + "_r" + repetition + "_"
							+ RunConfig.getTimestampNow() + ".txt";
					return ArcadeMachine.runOneGame(RunConfig
							.getGamePath(gameLevelPair.game), RunConfig
							.getGameLevelPath(gameLevelPair.game, level), false,
							config.getController(),
							(config.isSaveActions()) ? actionsFile : null,
							new Random().nextInt());
				}
			}
		} 
		return 0;
	}

	/**
	 * Run the configured games with the configured controller without visual
	 * feedback.
	 * 
	 * @param config
	 *            The run configuration containing the game details.
	 */
	public static void runGames(RunConfig config) {
		for (GameLevelPair<String, String[]> gameLevelPair : config.getGameLevels()) {
			ArcadeMachine.runGames(RunConfig.getGamePath(gameLevelPair.game),
					RunConfig.getGameLevelPaths(gameLevelPair.game,
							gameLevelPair.level), config.getRepetitions(),
					config.getController(), config.getRecordingPathsForGame(
							gameLevelPair.game, gameLevelPair.level));
		}
	}

	/**
	 * Run the configured games and play them yourself
	 * 
	 * @param config
	 *            The run configuration containing the game details.
	 */
	public static void playGamesYourself(RunConfig config) {
		for (GameLevelPair<String, String[]> gameLevelPair : config
				.getGameLevels()) {
			for (String level : gameLevelPair.level) {
				for (int repetition = 0; repetition < config.getRepetitions(); repetition++) {
					String actionsFile = "actions_game_" + gameLevelPair.game
							+ "_lvl_" + level + "_r" + repetition + "_"
							+ RunConfig.getTimestampNow() + ".txt";
					ArcadeMachine.playOneGame(RunConfig
							.getGamePath(gameLevelPair.game), RunConfig
							.getGameLevelPath(gameLevelPair.game, level),
							(config.isSaveActions()) ? actionsFile : null,
							new Random().nextInt());
				}
			}

		}
	}

	/**
	 * Replay a recorded game
	 * 
	 * @param readActionsFile
	 *            The file name of the recorded game.
	 */
	public static void replayGame(String readActionsFile) {
		String[] split = readActionsFile.split("_");
		ArcadeMachine.replayGame(RunConfig.getGamePath(split[2]),
				RunConfig.getGameLevelPath(split[2], split[4]), true,
				readActionsFile);
	}

}