from policies import base_policy as bp
import numpy as np
from policies import base_policy as bp
from math import floor, ceil

EPSILON = 0.5
ACTIONS = ["L", "F", "R"]

PLAYBACK_FINAL_ROUND = 1000

max_same = {'F': 40, 'L': 10, 'R': 10}
same_counter = 0

PADDING = 2
RADIUS_ENV = PADDING
OBJ_N = 11
density_radius_ext = 5


hd_epsilon_0=0.5
hd_eta_0=0.01
hd_eta_decay_param=0.000001
hd_eta_decay="Fixed"
hd_epsilon_decay="exp"
hd_epsilon_decay_param=0.0000003
hd_total_rounds_number=5000
hd_final_epsilon=0
hd_final_eta=0
hd_features_selected=["board_env","eaten_obj"]
hd_gamma=0.90


class Linear205999063(bp.Policy):
	"""
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentage of actions which are randomly chosen.
    """

	def cast_string_args(self, policy_args):
		return policy_args

	def init_run(self):

		self.epsilon_0 = hd_epsilon_0
		self.eta_0 = hd_eta_0
		self.eta_decay_param = hd_eta_decay_param
		self.eta_decay = hd_eta_decay
		self.epsilon_decay = hd_epsilon_decay
		self.epsilon_decay_param = hd_epsilon_decay_param
		self.total_rounds_number = hd_total_rounds_number
		self.final_epsilon = hd_final_epsilon
		self.final_eta = hd_final_eta
		self.features_selected = hd_features_selected
		self.gamma = hd_gamma

		self.states_queue = []
		self.learn_entrance = 1
		self.read_state = 0
		self.features_builder = FeaturesBuilder(self.features_selected)
		self.phi_length = self.features_builder.length_phi
		self.weights = np.random.normal(scale=0.001, size=self.phi_length)
		self.r_sum = 0

		self.epsilon = self.epsilon_0
		self.eta = self.eta_0

	def q_at_st(self, phi):
		"""
		Return the q-function for vector phi, given weights
		:param phi: A feature vector
		:return: Dot product of the weights and features
		"""
		return np.dot(phi, self.weights)

	def max_q(self, new_state, round_):
		"""
		Calculate for the given state the best action to perform according
		to q-function
		:param state_: state we want to evaluate
		:param round_: the round we got the state
		:return: the action that reaches the maximal q-value for the given
		state, and the q-value itself
		"""
		scores_ = []
		policies_lst = list(np.random.permutation(bp.Policy.ACTIONS))

		for a in policies_lst:
			phi = self.features_builder.build_features_vector(new_state, a,
															  round_)
			scores_.append(self.q_at_st(phi))
		return policies_lst[np.argmax(scores_)], max(scores_)

	def learn_single(self, round_game, prev_state, prev_action, reward,
					 queue_location, new_state):
		"""
		Update the weights by learning a single state (board + action + reward)
		:param round_game: the round the state was given
		:param prev_state: the state to learn
		:param prev_action: the action to learn
		:param reward: the reward we got for the action and the state
		:param queue_location: location in the queue of states
		:param new_state: the state the board reaches after performing
		action on prev state
		:return: None. Juste updating the weights vector
		"""
		phi = self.features_builder.build_features_vector(prev_state,
														  prev_action,
														  round_game)
		q_ = self.q_at_st(phi)
		max_future_act = self.max_q(new_state, round_game)[1]
		delta_ = q_ - (reward + self.gamma * max_future_act)
		self.weights = self.weights - self.eta * delta_ * phi

	def learn(self, round, prev_state, prev_action, reward, new_state,
			  too_slow):

		"""
		The process of learning
		Update the learning rate eta
		For each of the state in the queue, update the weights vector

		The arguments aren't interesting since we don't learn the last state specifically

		Sum the score for scoring

		:param round: the round of the game
		:param prev_state: prev_state
		:param prev_action: prev_action
		:param reward: reward
		:param new_state: new_state
		:param too_slow:
		:return:
		"""
		self.update_eta(self.learn_entrance)
		self.learn_entrance += 1
		while self.read_state < len(self.states_queue):
			round, prev_state, prev_action, reward, new_state = \
				self.states_queue[self.read_state]
			self.learn_single(round, prev_state, prev_action, reward,
							  self.read_state, new_state)
			self.read_state += 1

		try:
			if round % 100 == 0:
				if round > self.game_duration - self.score_scope:
					self.log(
						"Rewards in last 100 rounds which counts towards the score: " + str(
							self.r_sum), 'VALUE')
				else:
					self.log("Rewards in last 100 rounds: " + str(self.r_sum),
							 'VALUE')
				self.r_sum = 0
			else:
				self.r_sum += reward

		except Exception as e:
			self.log("Something Went Wrong...", 'EXCEPTION')
			self.log(e, 'EXCEPTION')

	def update_eta(self, round_):
		"""
		Update eta according to the current round of the game
		According to the option choosed (Fixed, Linear, Exp)
		:param round_: The round
		:return:
		"""
		if self.eta_decay == "Fixed":
			self.eta = self.eta_0
		elif self.eta_decay == "Linear":
			self.eta = (round_ - 1) * (self.final_eta - self.eta_0) / (
					self.total_rounds_number - PLAYBACK_FINAL_ROUND - 1) - self.eta_0
		else:
			self.eta *= (1. / (1. + self.eta_decay_param * round_))

	def update_epsilon(self, round_):
		"""
		Update epsilon according to the current round of the game
		According to the option choosed (Fixed, Linear, Exp)

		Note that in the 1000 last rounds, we wamt epsilon to be 0. (No
		exploration)

		In the exp option, we lower bound the value of epsilon so we always
		explore in a certain rate

		:param round_: The round
		:return:
		"""

		if round_ > self.total_rounds_number - PLAYBACK_FINAL_ROUND:
			self.epsilon = 0
		elif self.epsilon_decay == "Fixed":
			self.epsilon = self.epsilon_0
		elif self.epsilon_decay == "Linear":
			m_ = (self.final_epsilon - self.epsilon_0) / (
					self.total_rounds_number - PLAYBACK_FINAL_ROUND - 1)
			self.epsilon = (round_ - 1) * m_ - self.epsilon_0
		else:
			self.epsilon *= (1. / (1. + self.epsilon_decay_param * round_))
			self.epsilon = max(0.01, self.epsilon)

	def act(self, round_, prev_state, prev_action, reward, new_state,too_slow):
		self.update_epsilon(round_)
		"""
		Overall process of act

		Update epsilon rate
		Add to queue of states the (prev_state, prev_action, reward)

		Choose whether to exploit or to explore
		If exploit:
		    Choose the action that reaches the maximum q-value
		If explore:
		    Choose a random actiom
		"""
		if round_ > 0:
			self.states_queue.append(
				[round_, prev_state, prev_action, reward, new_state])

		if np.random.rand() < self.epsilon:
			act_chosen = np.random.choice(bp.Policy.ACTIONS)
		else:
			act_chosen, _ = self.max_q(new_state, round_)
		return act_chosen


class FeaturesBuilder():
    # Dictionary that maps each feature to the length of the
    # representation of the feature
    # It permits to "init" a features_vector in an easy way. we create a
    # combination of the features
    def __init__(self, features_selected):
        """
        Init object
        Calculate the length of features
        Calculate a first matrix of distances

        :param features_selected: The features to use
        """
        # Dictionnary of functions for features
        # Each key corresponds to an optional feature
        self.fndict = {"immediate_env": self.immediate_environment,
                       "eaten_obj": self.eaten_object,
                       "board_env": self.crop_board,
                       "closest_dist": self.closest_distance_to_head,
                       "density": self.count_by_squares_,
                       "change_dens": self.change_in_density}
        self.previous_board = None
        self.previous_head_pos = None
        self.previous_direction = None
        self.previous_board_rot = None
        self.previous_head_rot = None
        self.previous_direction_rot = None
        self.previous_board_shape = None
        self.previous_action = None
        self.previous_board_rot_extended = None
        self.previous_board_shape_rot_extended = None
        self.previous_head_rot_extended = None
        self.features_selected = features_selected
        self.calculate_phi_length()
        self.density_radius = 5
        self.loophole_radius = 2
        self.snake_obj = None
        self.head_position_eternal = [PADDING, PADDING]
        self.centered_distance_matrix_x, self.centered_distance_matrix_y, self.centered_distance_matrix_sum = self.calculate_center_distance_matrix(
            250, 250)
        self.middle_x = int(floor(250 / 2))
        self.middle_y = int(floor(250 / 2))
        self.crop_init = False
        self.crop_indices_init()

    HEAD_DIRECTION = "U"
    DICT_INIT_DIRECT_N = {"U": "N", "R": "E", "L": "W"}
    DICT_INIT_DIRECT_E = {"L": "N", "R": "S", "U": "E"}
    DICT_INIT_DIRECT_S = {"U": "S", "R": "W", "L": "E"}
    DICT_INIT_DIRECT_W = {"L": "S", "R": "N", "U": "W"}

    DIRECTIONS_ACTION_DICT = [DICT_INIT_DIRECT_N, DICT_INIT_DIRECT_E,
                              DICT_INIT_DIRECT_S,
                              DICT_INIT_DIRECT_W]
    DICT_DIRECTIONS_INDEX = {"N": 0, "E": 1, "S": 2, "W": 3}
    ROTATION_BY_DIR = {"N": 0, "E": 3, "S": 2, "W": 1}

    features_length = {"immediate_env": OBJ_N * 3, "eaten_obj": OBJ_N * 3,
                       "board_env": OBJ_N * (8),
                       "closest_dist": OBJ_N, "density": OBJ_N * 3,
                       "loop": 1,
                       "change_dens": OBJ_N * 3}

    def immediate_environment(self):
        """
        We use the rotated board extended such that we can access easily
        each case of the board, regarding the position of the head

        Return 3 indicator vectors that are merged
        Each vector is an indicator of one of the position around the head
        (up, right, left) that indicate the object contained in each position
        :return: the vector indicator
        """
        base_ = np.zeros(OBJ_N * 3)
        idx_1 = self.previous_board_rot_extended[
            self.previous_head_rot_extended[0] - 1,
            self.previous_head_rot_extended[1]]
        idx_2 = self.previous_board_rot_extended[
            self.previous_head_rot_extended[0], self.previous_head_rot_extended[1] - 1]
        idx_3 = self.previous_board_rot_extended[
            self.previous_head_rot_extended[0], self.previous_head_rot_extended[1] + 1]
        base_[int(idx_1)] = 1
        base_[int(idx_2 + OBJ_N)] = 1

        base_[int(idx_3 + OBJ_N * 2)] = 1
        return base_

    def eaten_object(self):
        """
        A feature that coupled the action and the eaten obj
        So it's a vector indicator such that there is a 1 only in a single
        place: ACTION AND OBJECT TYPE
        :return: the vector of this feature
        """
        base_ = np.zeros(OBJ_N * 3)
        if self.previous_action == "F":
            head_after_move_0 = self.previous_head_rot_extended[0] - 1
            head_after_move_1 = self.previous_head_rot_extended[1]
        elif self.previous_action == "R":
            head_after_move_0 = self.previous_head_rot_extended[0]
            head_after_move_1 = self.previous_head_rot_extended[1] + 1
        else:
            head_after_move_0 = self.previous_head_rot_extended[0]
            head_after_move_1 = self.previous_head_rot_extended[1] - 1

        obj_id = self.previous_board_rot_extended[head_after_move_0, head_after_move_1]
        base_[int(bp.Policy.ACTIONS.index(self.previous_action) * OBJ_N + obj_id)] = 1
        return base_

    def crop_board_environment(self):
        """
        Crop around the head according to radius (from each side but not
        from bottom)
        Transform it to an indicator vector
        :return: The feature vector
        """
        head_x = self.previous_head_rot_extended[0]
        head_y = self.previous_head_rot_extended[1]
        board = self.previous_board_rot_extended
        cropped_M = board[head_x - RADIUS_ENV: head_x + 1, head_y - RADIUS_ENV:head_y + RADIUS_ENV + 1]
        flatten_cropped = cropped_M.flatten()
        b = np.zeros(flatten_cropped.shape[0] * OBJ_N)
        c = np.arange(flatten_cropped.shape[0]) * OBJ_N
        d = c + flatten_cropped
        b[d.astype(int)] = 1
        return b

    def calculate_center_distance_matrix(self, x, y):
        """
        Given x and y, create a matrix of size x, y
        such that the center is 0 and that each coordinate contains the
        distance to the center of the matrix

        Flatten the matrix
        :param x: the axis=0 of the matrix
        :param y: the axis=1 of the matrix
        :return: A flattened matrix of distances to the center
        """
        x_distances = self.calculate_distance_axis(int(floor(x / 2)), x, roll_false=False)
        y_distances = self.calculate_distance_axis(int(floor(y / 2)), y, roll_false=False)
        head_distances = np.add.outer(x_distances, y_distances).flatten()
        return x_distances, y_distances, head_distances

    def calculate_distance_axis(self, pos, length, roll_false=True):
        """
        Calculate a distance array from the "pos" of length "lenght"

        Create a vector:
        5 4 3 2 1 0 1 2 3 4 5

        :param pos: The "zero" position
        :param length: The length of the vector
        :param roll_false: If True, roll the vector
        :return:
        """
        l_ = length
        middle_ = int(floor(l_ / 2))
        l_zeros = np.zeros(l_)
        if l_ % 2 == 0:
            dist_1 = np.arange(0, middle_)
        else:
            dist_1 = np.arange(0, middle_ + 1)
        l_zeros[middle_:] = dist_1
        if l_ % 2 == 1:
            l_zeros[:middle_ + 1] = np.flipud(dist_1)
        if l_ % 2 == 0:
            l_zeros[:middle_ + 1] = np.flipud(np.arange(0, middle_ + 1))

        if roll_false == False:
            return l_zeros

        return np.roll(l_zeros, pos - middle_)

    def get_closest_obj_positions(self):
        """
        For each object type, calculate the distance to the head of the
        closest object

        Create matrix of distances to the head position
        Sort the matrix
        For each type object, get the occurence with the smallest distance
        to the head position

        Return 1/distance

        :return: the "similarity" distances
        """

        head_distances = np.roll(self.head_distances_basic, self.previous_head_pos[0], axis=0)
        head_distances = np.roll(head_distances, self.previous_head_pos[1], axis=1).flatten()
        head_distances[np.where(head_distances == 0)[0]] = 10000000
        dist_sorted_indices = head_distances.argsort()
        sorted_board = self.previous_board.flatten()[dist_sorted_indices]
        unique_vals, unique_indices = np.unique(sorted_board, return_index=True)
        unique_indices = dist_sorted_indices[unique_indices]

        # Create a vector of number of objects and replace the zeros
        distances = np.ones(OBJ_N) * ((np.sum(self.previous_board.shape) / 2) + 1)
        distances[unique_vals.astype(int)] = head_distances[unique_indices]
        return 1 / distances

    def crop_board_to_size(self):
        """
        In the init function we create a board of size 250*250 for the
        distances matrix
        Now we want to resize it to the real board size.
        :return: A matrix of distances to the center. The matrix size is as
        the board size
        """
        upper_limit = ceil((self.previous_board_shape[0] + 1) / 2)
        vect_x = np.zeros(self.previous_board_shape[0])
        vect_x[:upper_limit] = np.arange(upper_limit)
        if self.previous_board_shape[0] % 2 == 0:
            vect_x[upper_limit - 1:] = np.arange(1, upper_limit)[::-1]
        else:
            vect_x[upper_limit:] = np.arange(1, upper_limit)[::-1]

        upper_limit = ceil((self.previous_board_shape[1] + 1) / 2)
        vect_y = np.zeros(self.previous_board_shape[1])
        vect_y[:upper_limit] = np.arange(upper_limit)
        if self.previous_board_shape[0] % 2 == 0:
            vect_y[upper_limit - 1:] = np.arange(1, upper_limit)[::-1]
        else:
            vect_y[upper_limit:] = np.arange(1, upper_limit)[::-1]

        self.head_distances_basic = np.add.outer(vect_x, vect_y)

    def closest_distance_to_head(self):
        """
        For each object type, calculate the distance to the head of the
        closest object

        Create matrix of distances to the head position
        Sort the matrix
        For each type object, get the occurence with the smallest distance
        to the head position

        Return 1/distance
        """

        closest_positions = self.get_closest_obj_positions()
        return closest_positions

    def preprocess_state(self):
        """
        Preprocess state
        1. Board += 1
        2. Rotate board
        3. Extend the matrix of the board by adding padding to the sides
        """
        self.previous_board += 1
        self.rotate_state()
        self.extend_state()

    def change_in_density(self):
        """
        For each of the possible actions, calculate the change of density
        for each of the object by taking each direction
        We calculate the density in "density_radius"
        :return: the feature vector
        """
        board = self.previous_board_rot_extended
        head_position = self.previous_head_rot_extended
        action_ = self.previous_action

        #        if action_ == "F":
        line_of_interest_F1 = board[(head_position[0] + density_radius_ext + 1) % board.shape[0], :]
        line_of_interest_F2 = board[(head_position[0] + 1) % board.shape[0], :]
        #        if action_ == "R":
        line_of_interest_R1 = board[:, (head_position[1] + density_radius_ext + 1) % board.shape[1]]
        line_of_interest_R2 = board[:, (head_position[1] + 1) % board.shape[1]]
        #        else:
        line_of_interest_L1 = board[:, (head_position[1] - density_radius_ext - 1) % board.shape[1]]
        line_of_interest_L2 = board[:, (head_position[1] - 1) % board.shape[1]]

        lines_interest = [[line_of_interest_F1, line_of_interest_F2],
                          [line_of_interest_R1, line_of_interest_R2],
                          [line_of_interest_L1, line_of_interest_L2]]
        density_change = np.empty(OBJ_N * 3)
        for j in range(3):
            line_j = lines_interest[j]
            for i in range(OBJ_N):
                count_ = np.count_nonzero(line_j[0] == i) - np.count_nonzero(line_j[1] == i)
                idx_ = j * OBJ_N + i
                density_change[idx_] = count_
        return density_change

    def extend_state(self):
        """
        Take the maximum over (PADDING, density_radius_ext)
        Create a matrix PADDING + (board)+ PADDING
        (For the 2 axis)
        For the right padding, put the left columns of the board
        For the upper padding, put the lowest lines of the board

        Calculate the new shapes
        Calculate the new related positions of the head

        :return:
        """
        padding_intern = max(PADDING, density_radius_ext)

        M_extended = np.zeros((self.previous_board_rot.shape[0] + 2 * padding_intern,
                               self.previous_board_rot.shape[1] + 2 * padding_intern))
        M_extended[:padding_intern, padding_intern:-padding_intern] = self.previous_board_rot[-padding_intern:, :]
        M_extended[-padding_intern:, padding_intern:-padding_intern] = self.previous_board_rot[:padding_intern, :]
        M_extended[padding_intern:-padding_intern, : padding_intern] = self.previous_board_rot[:, -padding_intern:]
        M_extended[padding_intern:-padding_intern, -padding_intern:] = self.previous_board_rot[:, :padding_intern]
        M_extended[-padding_intern:, -padding_intern:] = self.previous_board_rot[:padding_intern, :padding_intern]
        M_extended[:padding_intern, :padding_intern] = self.previous_board_rot[-padding_intern:, -padding_intern:]
        M_extended[:padding_intern, -padding_intern:] = self.previous_board_rot[-padding_intern:, :padding_intern]
        M_extended[-padding_intern:, :padding_intern] = self.previous_board_rot[:padding_intern, -padding_intern:]
        M_extended[padding_intern:-padding_intern, padding_intern:-padding_intern] = self.previous_board_rot

        self.previous_board_rot_extended = M_extended
        self.previous_head_rot_extended = self.previous_head_rot + padding_intern
        self.previous_board_shape_rot_extended = M_extended.shape

    def rotate_state(self):
        """
        Rotate the board such that the head is always in the Northen direction
        Rotate also the head
        :return:
        """
        rot_num = self.ROTATION_BY_DIR[self.previous_direction]
        board_rotated = np.rot90(self.previous_board, rot_num)
        self.shape_rotated = board_rotated.shape
        h_x, h_y = self.previous_head_pos
        b_x, b_y = self.previous_board_shape
        if rot_num == 3:
            rot_head_loc = [h_y, b_x - (h_x + 1)]
        elif rot_num == 2:
            rot_head_loc = [b_x - (h_x + 1), b_y - (h_y + 1)]
        elif rot_num == 1:
            rot_head_loc = [b_y - (h_y + 1), h_x]
        else:
            rot_head_loc = [h_x, h_y]

        self.previous_board_rot = board_rotated
        self.previous_head_rot = np.array(rot_head_loc)
        self.previous_direction_rot = "N"

    def assign_to_self(self, state, action, round):
        """
        Given new state and action and round, assign to self object each
        argument so we can use them to build features vectors
        :param state:
        :param action:
        :param round:
        :return:
        """
        self.previous_head_pos = np.array(state)[1][0].pos
        self.previous_board = state[0].copy()
        self.previous_direction = state[1][1]
        self.previous_board_shape = self.previous_board.shape
        self.previous_action = action
        if not self.crop_init:
            self.crop_init = True
            self.crop_board_to_size()

    def select_features(self):
        """
        Call to each of the functions in features selected and merge all
        the created features vectors
        :return:
        """
        return np.concatenate([self.fndict[k]() for k in self.features_selected])

    def build_features_vector(self, state, action, round):
        """
        Process of building features vectors for a given state and action

        Preprocess the state
        Calculate each of the features vectors

        :param state:
        :param action:
        :param round:
        :return: The whole features vectors
        """
        self.assign_to_self(state, action, round)
        self.preprocess_state()
        return self.select_features()

    def calculate_phi_length(self):
        """
        Calculate the length of the features vector
        This is done only during the all game
        :return: Assign to self.length_phi
        """
        self.length_phi = sum([self.features_length[item] for item in self.features_selected])

    def crop_indices_init(self, padding_default=True):
        """
        Assign relative coordinates to the head position for calculating
        crop board.

        It is all the cases that have a maximum distance of 2 (or 3) to the
        head position
        :param padding_default: If True; padding = 2
        Else padding = 3
        :return:
        """
        if padding_default:
            list_indices = [
                [0, -2],
                [0, -1],
                [0, 1],
                [0, 2],
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [-2, 0]
            ]
        else:
            list_indices = [
                [0, -3],
                [0, -2],
                [0, -1],
                [0, 1],
                [0, 2],
                [0, 3],
                [-1, -2],
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [-1, 2],
                [-2, -1],
                [-2, 0],
                [-2, 1],
                [-3, 0],
            ]

        self.crop_indices = np.array(list_indices)

    def crop_board(self):
        """
        Create indicator vector of the cases that are around the head
        (According to the PADDING - taking a radius, not only 3 cases)
        :return: the features vector
        """
        temp_0 = (np.array(self.crop_indices) + self.previous_head_rot_extended)[:, 0]
        temp_1 = (np.array(self.crop_indices) + self.previous_head_rot_extended)[:, 1]
        indices_crop = self.previous_board_rot_extended[temp_0, temp_1]
        b = np.zeros(indices_crop.shape[0] * OBJ_N)
        c = np.arange(indices_crop.shape[0]) * OBJ_N
        d = c + indices_crop
        b[d.astype(int)] = 1
        return b

    def count_by_squares_(self):
        """
        For each direction (R, L, U) we calculate the number of each object
        in RADIUS environment
        :return: the features object
        """
        head_position = self.previous_head_rot_extended
        M_extended = self.previous_board_rot_extended
        radius_ = self.density_radius
        head_extended = np.array(head_position) + radius_

        square_1 = M_extended[head_extended[0] - radius_ + 1: head_extended[0] + 1,
                   head_extended[1] - floor(radius_ / 2) + 1:head_extended[1] + floor(radius_ / 2) + 1]

        square_2 = M_extended[head_extended[0] - floor(radius_ / 2) + 1: head_extended[
                                                                             0] + floor(radius_ / 2) + 1 + 1,
                   head_extended[1]:head_extended[1] + radius_]

        square_3 = M_extended[head_extended[0] - floor(radius_ / 2) + 1: head_extended[0] + floor(radius_ / 2) + 1 + 1,
                   head_extended[1] - radius_:head_extended[1]]
        squares = [square_1, square_2, square_3]
        all_squares_counts = np.zeros(OBJ_N * 3)
        counter_sq = 0
        for sq_ in squares:
            un_, counts_ = np.unique(sq_, return_counts=True)
            occurences = np.zeros(OBJ_N)
            occurences[un_.astype(int) + 1] = counts_
            all_squares_counts[
            counter_sq * OBJ_N:(counter_sq + 1) * OBJ_N] = occurences
            counter_sq += 1
        return all_squares_counts
