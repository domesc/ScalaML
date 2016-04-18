package regression

import breeze.linalg._

/**
 * Created by domesc on 03/04/16.
 */
class LinearModel{

  def gradientDescent(X: DenseMatrix[Double],
                      y: DenseVector[Double],
                      theta: DenseVector[Double],
                      alpha: Double,
                      num_iters: Int): (DenseVector[Double], DenseVector[Double])= {
    val m: Int = y.length
    val history: DenseVector[Double] = DenseVector.zeros(num_iters)

    @annotation.tailrec
    def descend(theta: DenseVector[Double],
                history: DenseVector[Double],
                remaining: Int): (DenseVector[Double], DenseVector[Double]) = remaining match {

      case 0 => (theta, history)
      case _ => {
        val updatedCost = alpha * (1.0/m) * (((X * theta) - y).t * X).t;
        val updatedTheta = theta - updatedCost
        history(remaining-1) = cost(X, y, updatedTheta);
        descend(updatedTheta, history, remaining - 1)
      }
    }

    descend(theta, history, num_iters)
  }

  def cost(X: DenseMatrix[Double], y: DenseVector[Double], theta: DenseVector[Double]): Double = {
    val m: Int = y.length
    sum((X * theta - y) :^ 2d) / (2 * m)
  }

}
