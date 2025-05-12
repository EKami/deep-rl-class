from python import Python
from python import PythonObject
from layout import Layout, print_layout


fn greedy_policy[
    T: Copyable & Movable
](q_table: List[List[T]], state: Int) raises -> List[T]:
    # Exploitation: take the action with the highest state, action value
    vect = q_table[state]
    action = argmax(vect).cast(List[T])
    return action


def epsilon_greedy_policy[
    T: Copyable & Movable
](env: PythonObject, q_table: List[List[T]], state: T, epsilon: Float64) -> Int:
    np = Python.import_module("numpy")
    # Randomly generate a number between 0 and 1
    random_num = np.random.rand()
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = greedy_policy(q_table, state)
    # else --> exploration
    else:
        action = Int(env.action_space.sample())

    return action


def train[
    T: Copyable & Movable
](
    n_training_episodes: Int,
    min_epsilon: Float64,
    max_epsilon: Float64,
    decay_rate: Float64,
    env: PythonObject,
    max_steps: Int,
    mut q_table: List[List[T]],
    gamma: Float64,
    learning_rate: Float64,
) -> List[List[T]]:
    np = Python.import_module("numpy")
    tqdm = Python.import_module("tqdm")
    progress_bar = tqdm.tqdm(total=n_training_episodes)

    for episode in range(n_training_episodes):
        progress_bar.update(1)
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * episode
        )
        # Reset the environment
        state, info = Python.tuple(env.reset())
        step = 0
        terminated = False
        truncated = False

        # repeat
        for step in range(max_steps):
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(env, Qtable, state, epsilon)

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated, truncated, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            difference = (
                reward
                + gamma * np.max(q_table[new_state])
                - q_table[state][action]
            )
            Qtable[state][action] = (
                q_table[state][action] + learning_rate * difference
            )

            # If terminated or truncated finish the episode
            if terminated or truncated:
                break

            # Our next state is the new state
            state = new_state
    return q_table


# TODO: Use https://docs.modular.com/mojo/manual/layout/layouts
fn initialize_q_table[
    T: Copyable & Movable
](state_space: Int, action_space: Int, default_val: T) -> List[List[T]]:
    q_table = List[List[T]]()
    for _ in range(state_space):
        # For each state, create a new list to represent the actions.
        row = List[T]()
        # Initialize this row with 'action_space' number of zeros.
        # The resize() method here expands the list to 'action_space' elements
        # and fills new elements with the provided value (0 in this case).
        row.resize(action_space, default_val)
        # Add the newly created row (a list of zeros) to our main q_table.
        q_table.append(row)
    return q_table


def frozenlake_q_learning():
    env_id = "FrozenLake-v1"
    gym = Python.import_module("gymnasium")
    env = gym.make(
        env_id, map_name="4x4", is_slippery=False, render_mode="rgb_array"
    )
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space", env.observation_space)
    print("Sample observation", env.observation_space.sample())
    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape", env.action_space.n)
    print("Action Space Sample", env.action_space.sample())

    state_space = Int(env.observation_space.n)
    print("Type of", env.observation_space.n)
    print("There are ", state_space, " possible states")

    action_space = Int(env.action_space.n)
    print("There are ", action_space, " possible actions")

    # Training parameters
    n_training_episodes = 10000  # Total training episodes
    learning_rate = 0.7  # Learning rate

    # Evaluation parameters
    n_eval_episodes = 100  # Total number of test episodes

    # Environment parameters
    max_steps = 99  # Max steps per episode
    gamma = 0.95  # Discounting rate
    eval_seed = []  # The evaluation seed of the environment

    # Exploration parameters
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.05  # Minimum exploration probability
    decay_rate = 0.0005  # Exponential decay rate for exploration prob

    Qtable_frozenlake = initialize_q_table[Float32](
        state_space, action_space, 0.0
    )
    Qtable_frozenlake = train(
        n_training_episodes,
        min_epsilon,
        max_epsilon,
        decay_rate,
        env,
        max_steps,
        Qtable_frozenlake,
        gamma,
        learning_rate,
    )


def main():
    frozenlake_q_learning()
