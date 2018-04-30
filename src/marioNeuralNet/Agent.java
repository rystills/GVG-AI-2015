package ch.idsia.agents;

import ch.idsia.benchmark.mario.environments.Environment;

/**
 * Created by IntelliJ IDEA.
 * User: Sergey Karakovskiy
 * Date: Mar 28, 2009
 * Time: 8:46:42 PM
 * package ch.idsia.controllers.agents;
 */
public interface Agent
{
    boolean[] getAction();

    void integrateObservation(Environment environment);

    void giveIntermediateReward(float intermediateReward);

    /**
     * clears all dynamic data, such as hidden layers in recurrent networks
     * just implement an empty method for a reactive controller
     */
    public void reset();

    public String getName();

    public void setName(String name);
}
