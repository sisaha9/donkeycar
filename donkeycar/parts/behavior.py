class BehaviorPart(object):
    '''
    Keep a list of states, and an active state. Keep track of switching.
    And return active state information.
    '''
    def __init__(self, states):
        '''
        expects a list of strings to enumerate state
        '''
        print("bvh states:", states)
        self.states = states
        self.active_state = 0
        self.one_hot_state_array = []
        for i in range(len(states)):
            self.one_hot_state_array.append(0.0)
        self.one_hot_state_array[0] = 1.0

    def increment_state(self):
        self.one_hot_state_array[self.active_state] = 0.0
        self.active_state += 1
        if self.active_state >= len(self.states):
            self.active_state = 0
        self.one_hot_state_array[self.active_state] = 1.0
        print("In State:", self.states[self.active_state])

    def decrement_state(self):
        self.one_hot_state_array[self.active_state] = 0.0
        self.active_state -= 1
        if self.active_state < 0:
            self.active_state = len(self.states) - 1
        self.one_hot_state_array[self.active_state] = 1.0
        print("In State:", self.states[self.active_state])

    def set_state(self, iState):
        self.one_hot_state_array[self.active_state] = 0.0
        self.active_state = iState
        self.one_hot_state_array[self.active_state] = 1.0
        print("In State:", self.states[self.active_state])

    def run(self):
        return self.active_state, self.states[self.active_state], self.one_hot_state_array

    def shutdown(self):
        pass

class PilotCondition:
    def run(self, mode):
        if mode == 'user':
            return False
        else:
            return True

# Choose what inputs should change the car.
class DriveMode:
    def __init__(self, ai_throttle_mult):
        self.ai_throttle_mult = ai_throttle_mult
    def run(self, mode,
            user_angle, user_throttle,
            pilot_angle, pilot_throttle):
        if mode == 'user':
            return user_angle, user_throttle
        elif mode == 'local_angle':
            return pilot_angle if pilot_angle else 0.0, user_throttle
        else:
            return pilot_angle if pilot_angle else 0.0, \
                    pilot_throttle * self.ai_throttle_mult \
                        if pilot_throttle else 0.0