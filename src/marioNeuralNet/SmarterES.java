package ch.idsia.evolution.ea;

import ch.idsia.agents.Agent;
import ch.idsia.agents.learning.SmarterMLPAgent;
import ch.idsia.benchmark.tasks.Task;
import ch.idsia.evolution.EA;
import ch.idsia.evolution.Evolvable;

public class SmarterES implements EA {

    private final Evolvable[] population; //list of all population members
    private final float[] fitness; //list of all members' fitness
    private final int elite; //number of population members which survive between generations (elite = p, population-elite = c)
    public final Task task; //evaluation task (simulation)
    private final int evaluationRepetitions = 1; //only need one evaluation to determine member score

    /**
     * construct a new SmarterES instance to oversee the evaluation and evolution of our population 
     * @param task: the evaluation task to use (fitness function)
     * @param initial: the very first (entirely untrained) instance of our NN agent
     * @param populationSize: the total size of the population
     * @param numParents: how many individuals in our population are parents (elites); the remaining slots are used for children
     */
    public SmarterES(Task task, Evolvable initial, int populationSize, int numParents) {
        //create our starting population, filled with instances of our starting agent
    	this.population = new Evolvable[populationSize];
        for (int i = 0; i < population.length; population[i] = initial.getNewInstance(), ++i);
        
        //create our starting fitness values (won't be filled in properly until the end of each trial)
        this.fitness = new float[populationSize];
        
        this.elite = numParents;
        this.task = task;
    }
    
    //satisfy EA implementation
    public Evolvable[] getBests() { return new Evolvable[]{ population[0]}; }
    public float[] getBestFitnesses() { return new float[]{ fitness[0]}; }

    /**
     * generate the next generation of children, using the specified mutation magnitude
     * @param mutationMagnitude: the mutation magnitude to use when mutating children (specify -1 to use current magnitude)
     */
    public void nextGeneration(float mutationMagnitude) {
    	//evaluate each of the elites (survivors of previous generation) to determine their fitness
        for (int i = 0; i < elite; evaluate(i), ++i);
        
        for (int i = elite; i < population.length; i++) {
        	//note that the worst of the elites will not be copied if elites > population.length/2
            population[i] = population[(i - elite)%elite].copy();
            SmarterMLPAgent sm = (SmarterMLPAgent)population[i];
            //chose two parents from the elite class to use for recombination (alternatively, choose elites in order)
            //
            int parent1Index = (int) (Math.random() * elite);
            int parent2Index = (int) (Math.random() * elite);
            sm.recombine((SmarterMLPAgent)population[parent1Index],(SmarterMLPAgent)population[parent2Index]);
            sm.mutate(mutationMagnitude);
            evaluate(i);
        }
        shuffle();
        sortPopulationByFitness();
    }
    
    /**
     * generate the next generation of children, using the current mutation magnitude
     */
    public void nextGeneration() {
    	nextGeneration(-1);
    }

    private void evaluate(int which) {
        fitness[which] = 0;
        for (int i = 0; i < evaluationRepetitions; i++)
        {
            population[which].reset();
            fitness[which] += task.evaluate((Agent) population[which])[0];
        }
        fitness[which] = fitness[which] / evaluationRepetitions;
    }

    /**
     * shuffle the population, swapping each member with another member chosen at random
     */
    private void shuffle() {
        for (int i = 0; i < population.length; i++) {
            swapPopAndFitness(i, (int) (Math.random() * population.length));
        }
    }

    /**
     * use a naive sort to sort the population from highest to lowest fitness
     */
    private void sortPopulationByFitness() {
        for (int i = 0; i < population.length; i++) {
            for (int j = i + 1; j < population.length; j++) {
                if (fitness[i] < fitness[j]) {
                	swapPopAndFitness(i, j);
                }
            }
        }
    }

    /**
     * swap the population members at the specified indices, along with their fitness values
     * @param i: the index of the first member
     * @param j: the index of the second member
     */
    private void swapPopAndFitness(int i, int j) {
    	//swap population
    	Evolvable gcache = population[i];
        population[i] = population[j];
        population[j] = gcache;
        //swap fitness
    	float cache = fitness[i];
        fitness[i] = fitness[j];
        fitness[j] = cache;
    }
}
