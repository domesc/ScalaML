package org.scalaml.regression

import java.io.File

import breeze.linalg._
import breeze.numerics.abs
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

/**
 * Created by domesc on 03/04/16.
 */
class LinearRegressionTest extends FlatSpec with Matchers with BeforeAndAfterAll {
  var model: LinearRegression = _
  var X: DenseMatrix[Double] = _
  var y: DenseVector[Double] = _

  override def beforeAll() = {
    model = LinearRegression()
    X = DenseMatrix((3.0, 1.0), (-1.0, -2.0), (6.3, 5.2))
    y = DenseVector(5.4, 2.3, 4.5)
  }

  it should "compute the cost" in {
    val resourcesPath = System.getProperty("user.dir") + "/src/test/resources"
    val data = csvread(new File(resourcesPath + "/data/test1.csv"))
    val y: DenseVector[Double] = data(::, 1)
    val m = y.length
    val X = DenseMatrix(DenseVector.ones[Double](m), data(::, 0))
    val theta: DenseVector[Double] = DenseVector.zeros(2)

    val cost = model.cost(X.t, y, theta, 0.0)
    assert(Math.abs(cost - 32.07) <= 1.0e-2)
  }

  it should "verify normal fitting" in {
    model.fit(X, y)
    val diffNoReg = abs(model.predict(X) - y)

    // check convergence and distance from target value
    model.costHistory(0) shouldBe <(model.costHistory(-1))
    all(diffNoReg.toArray) shouldBe <=(2.5e-1)
  }

  it should "verify fitting with regularization term" in {
    val newModel = model.copy(regParam = 1e-5)
    newModel.fit(X, y)
    val diffReg = abs(newModel.predict(X) - y)

    // check convergence and distance from target value
    newModel.costHistory(0) shouldBe <(newModel.costHistory(-1))
    all(diffReg.toArray) shouldBe <=(2.5e-1)
  }

  it should "verify fitting with bigger regularization term" in {
    val newModel = model.copy(regParam = 10)
    newModel.fit(X, y)
    val diffRegUF = abs(newModel.predict(X) - y)

    // check convergence and distance from target value (underfitting)
    newModel.costHistory(0) shouldBe <(newModel.costHistory(-1))
    atLeast(2, diffRegUF.toArray) shouldBe >(2.0)
  }
}
