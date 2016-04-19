package org.scalaml.regression

import java.io.File

import breeze.linalg._
import breeze.numerics.abs
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfter, FunSuite, Matchers}

/**
 * Created by domesc on 03/04/16.
 */
@RunWith(classOf[JUnitRunner])
class LinearRegressionTest extends FunSuite
  with Matchers
  with BeforeAndAfter{

  test("Compute cost test") {
    val resourcesPath = System.getProperty("user.dir") + "/src/test/resources"
    val data = csvread(new File(resourcesPath + "/data/test1.csv"))
    val y: DenseVector[Double] = data(::, 1)
    val m = y.length
    val X = DenseMatrix(DenseVector.ones[Double](m), data(::, 0))
    val theta: DenseVector[Double] = DenseVector.zeros(2)
    val model =  new LinearRegression()

    val cost = model.cost(X.t, y, theta, 0.0)
    assert(Math.abs(cost - 32.07) <= 1.0e-2)
  }

  test("Gradient descent test (with and without regularization)") {
    val X = DenseMatrix((3.0, 1.0), (-1.0, -2.0), (6.3, 5.2))
    val y = DenseVector(5.4, 2.3, 4.5)
    val alpha = 0.01
    val model =  new LinearRegression()

    // verify normal fitting
    model.fit(X, y)
    val diffNoReg = abs(model.predict(X) - y)

    // check convergence and distance from target value
    model.costHistory(0) shouldBe < (model.costHistory(-1))
    all(diffNoReg.toArray) shouldBe <= (2.5e-1)

    // fitting with regularization term
    model.fit(X, y, lambda = 1e-5)
    val diffReg = abs(model.predict(X) - y)

    // check convergence and distance from target value
    model.costHistory(0) shouldBe < (model.costHistory(-1))
    all(diffReg.toArray) shouldBe <= (2.5e-1)

    // fitting with regularization term
    model.fit(X, y, lambda = 10)
    val diffRegUF = abs(model.predict(X) - y)

    // check convergence and distance from target value (underfitting)
    model.costHistory(0) shouldBe < (model.costHistory(-1))
    atLeast(2, diffRegUF.toArray) shouldBe > (2.0)
  }




}
