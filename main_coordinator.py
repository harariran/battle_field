from warnings import filterwarnings
from utils import factory
import tests
from environments.env_wrapper import CreateEnvironment_Battle

# Just to silent an harmless warning
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

# mac_BF_env = factory.CreateEnvironment()

mac_BF_env = CreateEnvironment_Battle(minimap=False)


# tests.test_centralized_controller(mac_BF_env)

# tests.test_decentralized_controller(mac_BF_env)

# tests.test_sim_controller(mac_BF_env)

# tests.test_coordinator(mac_BF_env)

# tests.test_sim_coordinator(mac_BF_env)

# tests.test_sim_teams(mac_BF_env)

# tests.test_sim_teams2(mac_BF_env)

# tests.test_sim_teams3(mac_BF_env)

# tests.test_sim_teams4(mac_BF_env)

tests.test_sim_teams5(mac_BF_env)


