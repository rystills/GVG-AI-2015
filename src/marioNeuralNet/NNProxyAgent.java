package marioNeuralNet;

import core.game.StateObservation;
import core.player.AbstractPlayer;
import ontology.Types.ACTIONS;
import tools.ElapsedCpuTimer;

//my job is to act as a proxy and pass data along to whatever agent we wish to train, so that I may be piped directly into the GVGAI runner
public class NNProxyAgent extends AbstractPlayer {
	public static SmarterMLPAgent curAgent;
	
	//blank constructor to satisfy GVG Runner
	 public NNProxyAgent(StateObservation gameState, ElapsedCpuTimer elapsedTimer) { }

	@Override
	public ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
		return curAgent.act(stateObs, elapsedTimer);
	}

}
