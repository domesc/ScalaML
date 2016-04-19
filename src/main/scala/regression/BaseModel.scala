package regression

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by domesc on 19/04/16.
  */
trait BaseModel {
  def fit(alpha: Double,
          num_iters: Int,
          lambda: Double = 0.0): (DenseVector[Double], DenseVector[Double])

  def predict(X: DenseMatrix[Double], theta: DenseVector[Double]): DenseVector[Double] = X * theta
}
