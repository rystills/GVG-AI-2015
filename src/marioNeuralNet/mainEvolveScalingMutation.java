package main;

import ch.idsia.agents.Agent;
import ch.idsia.agents.learning.SmarterMLPAgent;
import ch.idsia.benchmark.mario.engine.GlobalOptions;
import ch.idsia.benchmark.tasks.ProgressTask;
import ch.idsia.evolution.Evolvable;
import ch.idsia.evolution.ea.SmarterES;
import ch.idsia.tools.CmdLineOptions;
import ch.idsia.utils.wox.serial.Easy;

public class mainEvolveScalingMutation {
    final static int generations = 30;

    public static void main(String[] args) {
        CmdLineOptions options = new CmdLineOptions(new String[0]);
        options.setPauseWorld(false);
        Evolvable initial = new SmarterMLPAgent();
        options.setFPS(GlobalOptions.MaxFPS);
        options.setLevelDifficulty(0); //change level difficulty
        options.setLevelRandSeed(2); //change level seed
        options.setVisualization(false);
        ProgressTask task = new ProgressTask(options); //defines fitness function
        SmarterES es = new SmarterES(task, initial, 50, 25); //50 total population, with 25 parents = 25 children (even split)
        System.out.println("Evolving " + initial + " with task " + task);   
        final String fileName = "evolved" + (int) (Math.random() * Integer.MAX_VALUE) + ".xml";
        
        boolean scalingMutation = true; //whether we should use a constant or scaling mutation
        float mutationMagnitude = .3f; //starting mutation magnitude, if using scaling mutation
        float mutationReductionRate = .003f; //rate at which our mutation magnitude should reduce after each generation
        
        for (int gen = 0; gen < generations; gen++) {
            es.nextGeneration(scalingMutation ? mutationMagnitude : -1);
            mutationMagnitude -= mutationReductionRate;
            double bestResult = es.getBestFitnesses()[0];
            System.out.println("Generation " + gen + " best " + bestResult);
            Easy.save(es.getBests()[0], fileName);
        }
        //run the final version of our evolved agent
		System.out.println("Showing results of final evolved agent with score: " + es.getBestFitnesses()[0]);
		options.setFPS(24);
		options.setVisualization(true);
        es.task.evaluate((Agent)es.getBests()[0]);
    }
}
