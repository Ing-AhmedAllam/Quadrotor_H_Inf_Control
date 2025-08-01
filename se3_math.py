import numpy as np

class se3_math():
    def __init__(self):
        pass
    
    def euler2dcm(self, roll, pitch, yaw):
        """
        Convert Euler angles (roll, pitch, yaw) to Direction Cosine Matrix (DCM).
        :param roll: Rotation around x-axis in radians
        :param pitch: Rotation around y-axis in radians
        :param yaw: Rotation around z-axis in radians
        :return: 3x3 Direction Cosine Matrix
        """
        
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        
        return R_z @ R_y @ R_x
    
    def dcm2euler(self, R):
        """
        Convert Direction Cosine Matrix (DCM) to Euler angles (roll, pitch, yaw).
        :param R: 3x3 Direction Cosine Matrix
        :return: roll, pitch, yaw in radians
        """
        if R[2, 0] < -1 or R[2, 0] > 1:
            raise ValueError("Invalid DCM: out of range for arcsin.")
        
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(-R[2, 0])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        
        return roll, pitch, yaw
    
    def quaternion2dcm(self, q):
        """
        Convert quaternion to Direction Cosine Matrix (DCM).
        :param q: Quaternion in the form [qx, qy, qz, qw]
        :return: 3x3 Direction Cosine Matrix
        """
        qw, qx, qy, qz = q
        
        R = np.array([[1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                      [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                      [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]]).reshape(3, 3)
        
        return R
    
    def dcm2quaternion(self, R):
        """
        Convert Direction Cosine Matrix (DCM) to quaternion.
        :param R: 3x3 Direction Cosine Matrix
        :return: Quaternion in the form [qx, qy, qz, qw]
        """
        q1 = np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
        div_4q1 = 1.0 / (4.0 * q1)
        q2 = (R[1, 0] + R[0, 1]) * div_4q1
        q3 = (R[2, 0] + R[0, 2]) * div_4q1
        q0 = (R[1, 2] - R[2, 1]) * -div_4q1
        quat = [q3, q0, q1, q2]

        return quat
    
    def vee_map_3x3(self, skew_symmetric_matrix):
        """
        Convert a skew-symmetric matrix to a 3D vector using the vee map.
        :param skew_symmetric_matrix: 3x3 skew-symmetric matrix
        :return: 3D vector
        """
        if skew_symmetric_matrix.shape != (3, 3):
            raise ValueError("Input must be a 3x3 skew-symmetric matrix.")
        if not np.allclose(skew_symmetric_matrix, -skew_symmetric_matrix.T, atol=1e-8):
            raise ValueError("Input must be skew-symmetric.")
        
        return np.array([skew_symmetric_matrix[2, 1], 
                         skew_symmetric_matrix[0, 2], 
                         skew_symmetric_matrix[1, 0]])
        
    def hat_map_3d(self, vector):
        """
        Convert a 3D vector to a skew-symmetric matrix using the hat map.
        :param vector: 3D vector
        :return: 3x3 skew-symmetric matrix
        """
        vector = np.array(vector).flatten()
        if len(vector) != 3:
            raise ValueError("Input must be a 3D vector.")

        return np.array([[0, -vector[2], vector[1]],
                         [vector[2], 0, -vector[0]],
                         [-vector[1], vector[0], 0]]).reshape(3, 3)

    def orthonormalize_dcm(self, R):
        """
        Ensure that a Direction Cosine Matrix (DCM) is orthonormal.
        :param R: 3x3 Direction Cosine Matrix
        :return: Orthonormalized DCM
        """
        assert R.shape == (3, 3), "Input must be a 3x3 matrix."
        # assert np.isclose(np.linalg.det(R), 1.0), "Input must be a proper rotation matrix with determinant close to 1."
        assert not np.isnan(R).any(), f"Input matrix contains NaN values\n {R}"
        assert not np.isinf(R).any(), f"Input matrix contains Inf values\n {R}"

        U, _, Vt = np.linalg.svd(R)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt
        return R_ortho
    
    def get_prv_angle(self, R):
        """
        Get the principal rotation vector (PRV) angle from a Direction Cosine Matrix (DCM).
        :param R: 3x3 Direction Cosine Matrix
        :return: Principal rotation angle in radians
        """
        if R.shape != (3, 3):
            raise ValueError("Input must be a 3x3 matrix.")

        if np.linalg.det(R) <= 0:
            raise ValueError("DCM must be a proper rotation matrix with determinant > 0.")

        cos_angle = (np.trace(R) - 1) / 2
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # prevent numerical errors
        angle = np.arccos(cos_angle)

        return angle

