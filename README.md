<h1>Policy gradient algorithms</h1>
<p>This repository contains two policy gradient RL algorithms applied to three games - rock paper scissors, 
<a href="https://github.com/rmurphy120/FictitiousPlay" target="_blank">the car game, and the soccer game</a>. These are
a class of algorithms which tries to find the optimal policy through machine learning. The general framework is that the
state is fed into a network which outputs a policy directly. Using the policy gradient theorem, you can learn an optimal
policy. <a href="https://lilianweng.github.io/posts/2018-04-08-policy-gradient/" target="_blank">This blog</a> is a 
great survey of policy gradient algorithms. It includes a lot of the theory behind these algorithms. Usually these algorithms
are framed within single agents problems, but this project was interested in the multi-agent formulation of these 
algorithms.</p>

<h3>REINFORCE</h3>
<p>This is the original policy gradient algorithm. It samples a sequence of states on-policy (Called a trajectory)
which is uses to estimate Q.</p>
<p>Below is a graph of the trajectory length and agents' expected reward throughout training in the car game (It was trained to 
where the trajectory would stop once an agent went off the board, so this shows them learning to stay on the board)</p>
<img src="Images/REINFORCECar.png" alt="REINFORCE Trajectory and exp reward over time" width=500 height=275>

<h3>A2C</h3>
<p>This is a foundational policy gradient algorithm that falls under the actor-critic paradigm which is featured in 
many policy gradient algorithms. In addition to networks being used to model the policy of the agents (Actors), an 
additional critic networks are used to estimate the value function of each player (One critic network can be used for 
both agents if 
the game is zero-sum or common‑payoff). Traditional A2C uses trajectories to estimate V to learn the actor and critic networks, 
but we altered it to take the expectation over all actions pairs to calculate V directly, thus better learning the critic network and 
eliminating randomness from the training (Besides initialization). One advantage of A2C is we can use the value network
to determine convergence of the system, where REINFORCE has an arbitrary cutoff.</p>
<p>Below is a graph of the policy losses and the value loss throughout training on the car game.</p>
<img src="Images/A2CCar.png" alt="A2C losses over time" width=500 height=275>

<h3>Performance on Rock Paper Scissors</h3>
<h4>REINFORCE</h4>
<img src="Images/REINFORCERPS.png" alt="REINFORCE policy over time" width=500 height=275>
<h4>A2C</h4>
<img src="Images/A2CRPS.png" alt="A2C policy over time" width=500 height=275>

<p>The above graphs plot the policy of each agent across training. The policies they produce 
have similar bounds, and in general they exhibit the same behavior, but A2C is 
noticably more smooth. The graphs give a good intuition how these algorithms are driven by best response 
dynamics - an agent that plays paper a lot causes the other agent to play scissors a lot and so on. It also illustraites
the difficulty these algorithms have with modeling mixed nash equilibrium.</p>

<h3>Performance on The Car and Soccer Games</h3>
<p>Both algorithms give subpar performance on these games (A2C not yet run on soccer game). Part of the issue is that
these algorithms are rewarded for being certain (e.i. outputting deterministic policies). These two games, however, have
many states where there is no deterministic Nash equilibrium, only mixed. So these policies will fail at these states, 
which have repercussive effects to other states that do have deterministic Nash equilibrium (Because every state affects 
neighboring states). The reason this doesn't happen in rock paper scissors is there isn't enough complexity to 
destabilize it.</p>