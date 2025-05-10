from python import Python
from python import PythonObject


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

    state_space = env.observation_space.n
    print("Type of", env.observation_space.n)
    print("There are ", state_space, " possible states")

    action_space = env.action_space.n
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

    # initialize_q_table[Float32](state_space, action_space, 0)


struct NameList:
    var names: List[String]

    def __init__(out self, *names: String):
        self.names = List[String]()
        for name in names:
            self.names.append(name[])

    def __getitem__(ref self, index: Int) -> String:
        if index >= 0 and index < len(self.names):
            return self.names[index]
        else:
            raise Error("index out of bounds")

    def __setitem__(mut self, index: Int, value: String):
        self.names[index] = value


trait Quackable:
    fn quack(self):
        ...


@value
struct Duck(Quackable):
    fn quack(self):
        print("Quack")


@value
struct Container[ElementType: Copyable & Movable]:
    var element: ElementType

    def __str__[
        StrElementType: Writable & Copyable & Movable, //
    ](self: Container[StrElementType]) -> String:
        return String(self.element)


def main():
    float_container = Container(5)
    string_container = Container("Hello")
    var small_vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    print("SIMD vector element type:", small_vec.element_type)

    print(small_vec)
    print(float_container.__str__())
    print(string_container.__str__())
