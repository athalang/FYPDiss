from quat import qgeodesic, qnorm, qdot

def geodesic_loss(q1, q2):
    return qgeodesic(qnorm(q1), qnorm(q2)).mean()

def directional_loss(q1, q2, l=1.0):
    q1 = qnorm(q1)
    q2 = qnorm(q2)
    return (qgeodesic(q1, q2) + l * (1 - qdot(q1, q2))).mean()