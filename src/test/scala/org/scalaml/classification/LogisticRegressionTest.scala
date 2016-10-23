package org.scalaml.classification

import java.io.File

import breeze.linalg.{DenseVector, _}
import org.scalatest.{BeforeAndAfter, FunSuite, Matchers}

/**
  * Created by domesc on 20/04/16.
  */
class LogisticRegressionTest extends FunSuite
  with Matchers
  with BeforeAndAfter{

  test("Compute cost test") {
    val resourcesPath = System.getProperty("user.dir") + "/src/test/resources"
    val data = csvread(new File(resourcesPath + "/data/test_log_reg.csv"))
    val y: DenseVector[Double] = data(::, 2)
    val m = y.length
    val X = DenseMatrix(DenseVector.ones[Double](m), data(::, 0), data(::, 1))
    val theta: DenseVector[Double] = DenseVector.zeros(3)
    val model =  new LogisticRegression()

    val cost = model.cost(X.t, y, theta, 0.0)
    assert(Math.abs(cost - 0.693) <= 1.0e-2)
  }

  test("Gradient descent test (with and without regularization)") {
  }
}
