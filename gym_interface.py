
class GymInterface:
    def __init__(self,space):
        self.space = space

    def toGym(self, s):
        """
        translate a standard vector to an openAI gym space sample
        """
    def toVec(self, s):
        """
        translate a openAI Gym space sample to a standard vector
        """
