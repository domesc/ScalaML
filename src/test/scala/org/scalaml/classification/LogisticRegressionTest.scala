package org.scalaml.classification

import java.io.File

import breeze.linalg.{ DenseVector, _ }
import org.scalatest._

/**
 * Created by domesc on 20/04/16.
 */
class LogisticRegressionTest extends FlatSpec
    with Matchers
    with BeforeAndAfterAll {

  var model: LogisticRegression = _
  var features: DenseMatrix[Double] = _
  var labels: DenseVector[Double] = _

  override def beforeAll(): Unit = {
    model = LogisticRegression()
    features = DenseMatrix((3.0, 1.0), (-1.0, -2.0), (6.3, 5.2), (3.0, 1.0), (4.3, 5.1), (6.3, 5.2))
    labels = DenseVector(0.0, 1.0, 1.0, 0.0, 1.0, 1.0)
  }

  it should "Compute cost test" in {
    val resourcesPath = System.getProperty("user.dir") + "/src/test/resources"
    val data = csvread(new File(resourcesPath + "/data/test_log_reg.csv"))
    val y: DenseVector[Double] = data(::, 2)
    val m = y.length
    val X = DenseMatrix(DenseVector.ones[Double](m), data(::, 0), data(::, 1))
    val theta: DenseVector[Double] = DenseVector.zeros(3)

    val cost = model.cost(X.t, y, theta, 0.0)
    assert(Math.abs(cost - 0.693) <= 1.0e-2)
  }

  it should "verify normal fitting" in {
    model.fit(features, labels)

    // check convergence and distance from target value
    model.costHistory(0) shouldBe <(model.costHistory(-1))
    model.predict(features).toArray shouldEqual Array(0.0, 0.0, 1.0, 0.0, 1.0, 1.0)
  }

  it should "verify fitting with regularization term" in {
    val newModel = model.copy(regParam = 1e-5)
    newModel.fit(features, labels)

    // check convergence and distance from target value
    newModel.costHistory(0) shouldBe <(newModel.costHistory(-1))
    newModel.predict(features).toArray shouldEqual Array(0.0, 0.0, 1.0, 0.0, 1.0, 1.0)
  }

  it should "verify fitting with bigger regularization term" in {
    val newModel = model.copy(regParam = 10)
    newModel.fit(features, labels)

    // check convergence and distance from target value (underfitting)
    newModel.costHistory(0) shouldBe <(newModel.costHistory(-1))
    newModel.predict(features).toArray shouldEqual Array(1.0, 0.0, 1.0, 1.0, 1.0, 1.0)
  }
}
