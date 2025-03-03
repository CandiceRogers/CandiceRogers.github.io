# Professional Self-Assessment

## The Journey from Web Developer to AI Application Engineer

My journey in computer science began over two decades ago with web development, but it's been through recent focused study and deliberate skill development that I've transformed into a specialist in AI application engineering. Building the Treasure Hunt Game for my ePortfolio represents a culmination of this evolution, combining front-end development expertise with newly strengthened capabilities in back-end architecture and database integration. This project exemplifies how I've systematically expanded my technical repertoire beyond my comfort zone of front-end and LLM integration to become a more versatile developer capable of building complete, intelligent applications truly from the ground up, utilizing foundational machine learning methodologies blended with modern application development.

## Expanding Technical Foundations

Throughout my educational and professional journey, I've cultivated a range of technical and interpersonal skills essential for success in today's rapidly evolving software engineering landscape.

**Collaboration in Team Environments** has been increasignly important to my development approach. While my contractor role often involves independent work, I've embraced structured development practices that facilitate collaboration, such as modular code design and version control workflows. These practices have served me well when integrating with development teams for larger extant AI projects, where my components needed to interface seamlessly with other developers' work and pre-existing systems. The ability to both lead and support within technical teams has become invaluable as AI projects increasingly require cross-disciplinary expertise.

**Effective Stakeholder Communication** has evolved from a supplementary skill to a core competency in my work. Building AI applications requires translating complex technical concepts into understandable terms for clients and users. My academic growth has refined my ability to bridge these worlds by creating thorough documentation, requirements specification and user guides that non-technical stakeholders can understand and act upon. This skill proved particularly valuable when developing custom AI solutions for small and indepentently owned businesses where managing expectations about capabilities and limitations was critical to project success.

**Data Structures and Algorithms** represent an area where my academic journey has most significantly enhanced my professional capabilities. While I previously relied on framework-provided structures, I've now developed a deeper understanding of algorithmic principles that allows me to make more informed implementation choices. This is evident in how I approached the hybrid AI architecture in the Treasure Hunt Game by recognizing when Q-learning needed supplementation with A* pathfinding to overcome specific limitations. This algorithmic thinking now influences how I approach all development tasks, from data processing to optimization.

**Software Engineering and Database Integration** skills have matured through structured study and practical application. Moving beyond client-side development, I've strengthened my ability to design complete application architectures with full stack development garnering appropriate data persistence layers. Learning to evaluate database technologies based on specific application needs, as demonstrated by my selection of SQLite over MongoDB for the Treasure Hunt Game, has made me more effective at designing sustainable, maintainable systems that simply work. These enhanced back-end capabilities complement my established front-end expertise, allowing me to develop full-stack solutions independently that remain lightweight and user-friendly.

**Security Considerations** have become integrated into my development process at every stage. Rather than treating security as an afterthought, I now implement secure coding practices and appropriate authentication measures from initial design through implementation. This security-minded approach reflects my understanding that modern applications, particularly those involving AI and deployed on the web, must protect both system integrity and user data.

This academic and portfolio development process has reinforced core professional values that guide my work: a commitment to continuous learning, emphasis on user-centered design, and the importance of ethical considerations in AI application development. These values, combined with my expanded technical skill set, position me to contribute meaningfully to organizations developing intelligent applications that deliver genuine value to users.

## The ePortfolio: A Holistic Demonstration of Capabilities

The three artifacts in my portfolio collectively demonstrate my ability to develop a complete, intelligent application from concept to implementation. The Software Design and Engineering enhancement showcases my ability to transform theoretical concepts into interactive experiences through thoughtful UI/UX design and game mechanics. The Algorithms and Data Structures enhancement demonstrates my capacity to identify algorithmic limitations and implement sophisticated solutions that combine complementary approaches. The Database enhancement reveals my understanding of data persistence requirements and ability to select and implement appropriate storage solutions.

Together, these artifacts tell the story of a developer who can envision complete applications, solve complex technical challenges, and deliver engaging user experiences. They represent not just isolated skills, but the integration of multiple disciplines into cohesive, functional software.

As you explore the technical artifacts of the enhanced Treasure Hunt Game that follow, you'll see how each enhancement builds upon the others to create a project that demonstrates the full spectrum of my capabilities—from front-end interactivity to intelligent algorithms to data persistence. This portfolio stands as evidence of my readiness to tackle sophisticated software engineering challenges in the rapidly evolving field of AI application development.

# Code Review: Initial Assessment and Enhancement Planning

Before undertaking the enhancement of the Treasure Hunt Game, I conducted a systematic code review to identify limitations in the original implementation and plan strategic improvements. This initial assessment was crucial for developing a comprehensive enhancement roadmap that would transform the basic Q-learning demonstration into a fully interactive game with sophisticated AI capabilities.

## Examining the Original Implementation

The original project consisted of three main Python components working together to implement a basic Q-learning agent:

1. **TreasureMaze.py** - Defined the environment as a maze matrix where 1.0 represented free cells and 0.0 represented obstacles
2. **GameExperience.py** - Handled the experience replay mechanism for training
3. **Main Jupyter notebook** - Tied these components together using a neural network built with Keras

The implementation focused solely on the AI learning process with no user interface or interaction. The environment state was visualized through basic matplotlib displays showing the maze, agent position, and path taken.

## Code Structure Analysis

Examining the original code revealed several areas that needed improvement. The neural network model was quite basic, and the implementation lacked proper error handling and logging mechanisms. The code also didn't follow modern Python best practices for class structure and documentation.

```python
def build_model(maze):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model
```

The training loop in the `qtrain()` function showed another area for improvement. The use of global variables and magic numbers didn't align with current software engineering best practices:

```python
def qtrain(model, maze, **opt):
    # exploration factor
    global epsilon 
    # number of epochs
    n_epoch = opt.get('n_epoch', 15000)
    # maximum memory to store episodes
    max_memory = opt.get('max_memory', 1000)
    # maximum data size for training
    data_size = opt.get('data_size', 50)
    
    # Start time
    start_time = datetime.datetime.now()
    
    # Construct environment/game from numpy array: maze
    qmaze = TreasureMaze(maze)
    
    # Initialize experience replay object
    experience = GameExperience(model, max_memory=max_memory)
```

## Algorithmic Limitations

Looking deeper into the algorithmic implementation, I found several limitations in the core Q-learning process. The exploration factor (epsilon) was handled as a global variable, making it difficult to implement more sophisticated exploration strategies. The fixed memory size also limited the system's ability to learn from longer sequences of actions.

```python
# Choose action based on exploration or exploitation
if np.random.rand() < epsilon:
    action = random.choice(qmaze.valid_actions())  # explore
else:
    action = np.argmax(experience.predict(envstate))  # exploit
```

The simple epsilon-greedy approach could be enhanced with more sophisticated exploration strategies like softmax action selection or uncertainty-based exploration. Additionally, the neural network structure was quite basic with very few layers, and wasn't optimized for spatial understanding — a crucial factor for effective maze navigation.

```python
# Neural network architecture
model = Sequential()
model.add(Dense(maze.size, input_shape=(maze.size,)))
model.add(PReLU())
model.add(Dense(maze.size))
model.add(PReLU())
model.add(Dense(num_actions))
model.compile(optimizer='adam', loss='mse')
```

## Data Management Limitations

The project also had significant data management limitations. The experience replay system in GameExperience.py managed training data using simple list structures with no persistence:

```python
# Stores episodes in memory
def remember(self, episode):
    # episode = [envstate, action, reward, envstate_next, game_over]
    # memory[i] = episode
    # envstate == flattened 1d maze cells info, including pirate cell (see method: observe)
    self.memory.append(episode)
    if len(self.memory) > self.max_memory:
        del self.memory[0]
```

The win history tracking was particularly basic, missing opportunities to store detailed performance data and learn from historical patterns:

```python
if game_status == 'win':
    win_history.append(1)
    game_over = True
elif game_status == 'lose':
    win_history.append(0)
    game_over = True
else:
    game_over = False
```

## Enhancement Planning

Based on this code review, I developed a comprehensive enhancement plan focused on three key areas:

### 1. Software Design and Engineering

The plan included creating a proper game engine structure around the existing Q-learning implementation:

```python
class TreasureHuntGame:
    def __init__(self):
        self.maze = TreasureMaze()  # Existing maze environment
        self.ai_agent = QLearnAgent()  # Existing AI agent
        self.player = Player()  # New player class
        self.game_mode = None  # 'AI' or 'Player' or 'Race'
    
    def initialize_game_modes():
        # Single player mode
        # AI demonstration mode
        # Race mode (Player vs AI)
```

This would involve separating the monolithic implementation into distinct components for game state management, visualization, and AI processing, while adding features like saving game states and comparing human versus AI performance.

### 2. Algorithms and Data Structures

The algorithmic enhancement would focus on implementing multiple pathfinding approaches and improving learning efficiency:

```python
class EnhancedTreasureHunt:
    def __init__(self):
        self.q_learner = QLearningAgent()  # Enhanced Q-learning
        self.a_star = AStarPathfinder()  # A* implementation
        self.dijkstra = DijkstraPathfinder()  # Dijkstra's algorithm
        self.analytics = PerformanceTracker()  # Performance monitoring
```

The simple action selection would be enhanced with uncertainty estimation:

```python
def select_action(self, state, uncertainty_threshold):
    q_values, uncertainty = self.model.predict_with_uncertainty(state)
    if uncertainty > uncertainty_threshold:
        return self.explore()
    else:
        return self.exploit(q_values)
```

### 3. Database Integration

The database enhancement would implement a comprehensive data management system:

```python
def store_game_state(self, state_data):
    self.db.game_states.insert_one({
        'timestamp': datetime.now(),
        'maze_config': state_data.maze,
        'player_position': state_data.position,
        'ai_state': state_data.ai_state,
        'metrics': {
            'moves': state_data.moves,
            'time': state_data.elapsed_time,
            'score': state_data.score
        }
    })
```

This initial code review was essential for identifying the specific limitations of the original implementation and developing a structured enhancement plan. By carefully examining the existing codebase I was able to target key areas for improvement that would transform the basic Q-learning demonstration into a fully-featured interactive game with more 
sophisticated AI capabilities and data persistence. The code review process demonstrated my ability to analyze existing software, identify technical limitations, and develop strategic enhancement plans that address those limitations while expanding functionality. This analytical approach set the foundation for the successful enhancements documented in the subsequent sections of this portfolio.

# Software Design and Engineering: Creating an Interactive Game Experience

The first enhancement of the Treasure Hunt project focused on transforming a simple Q-learning algorithm training exercise into a fully interactive game experience. This fundamental redesign expanded the application's scope far beyond its original purpose as an educational demonstration, creating a foundation for all subsequent enhancements.

## Transformation from Algorithm to Application

The original Treasure Hunt project existed solely as a Jupyter notebook using TensorFlow and Keras to demonstrate Q-learning concepts. It had no interactive elements, no user interface, and functioned purely as a technical demonstration of machine learning principles. My enhancement created an entirely new application architecture around this core, implementing:

- A complete game engine using the PyGame library
- Interactive user controls and visual feedback systems
- A significantly expanded maze environment beyond the original 8×8 grid
- Random maze generation using recursive division algorithms
- Player-versus-AI gameplay mechanics

## Design Philosophy and Implementation

I made deliberate design choices to balance innovation with the project's educational roots. While completely rebuilding the application's infrastructure, I maintained a visual style that connected to the original project—preserving its simplistic, utilitarian aesthetic rather than creating a graphically unrelated game:

![The refactored Treasure Hunt Game maze compared to original](/assets/images/comparison.png)

This design approach ensured that the enhanced project demonstrated clear evolution rather than complete replacement. The user interface clearly shows the maze environment, player position, AI agent, and includes features like:

- Clear visual distinction between walls, paths, and objectives
- Real-time visualization of both player and AI movement
- Intuitive controls for navigation and power-up interaction
- Status displays for game information and statistics

## Technical Implementation

The PyGame library provided the framework for this enhancement, allowing for efficient implementation of the graphical interface while maintaining Python as the core language. This implementation required:

1. Designing a modular code architecture that separated game logic from display functions
2. Creating responsive input handling for player controls
3. Implementing collision detection and boundary enforcement
4. Developing algorithms for procedural maze generation that guaranteed solvable puzzles
5. Building transition systems between game states (main menu, gameplay, end screens)

Working with PyGame expanded my technical capabilities, requiring me to learn new modules and techniques for rendering the maze, implementing recursive division for randomized layouts, and creating an intuitive user interface. This learning process deepened my understanding of game development principles while building on my existing programming knowledge.

## Enhancement Outcomes

This enhancement directly demonstrated my ability to "use well-founded and innovative techniques, skills, and tools in computing practices for the purpose of implementing computer solutions that deliver value." By transforming a technical demonstration into an interactive application, I showed how theoretical concepts can be made accessible and engaging through thoughtful design and implementation.

The enhanced application maintains the educational value of the original project while significantly expanding its utility and appeal. The implementation balances algorithmic complexity with user experience considerations, using industry-standard tools and frameworks including PyGame for the interface and numpy for efficient maze generation and manipulation.

Most importantly, this enhancement created the foundation upon which all subsequent improvements would build—establishing the core architecture that would later support advanced algorithmic integration and database functionality. By reimagining the project's fundamental purpose and structure, I created a platform that could showcase multiple dimensions of software engineering expertise.
