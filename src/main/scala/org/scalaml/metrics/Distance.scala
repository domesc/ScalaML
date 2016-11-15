package org.scalaml.metrics

import breeze.linalg._
import breeze.numerics.{ abs, pow }

/**
 * Various distance metrics, they can be used in several machine learning algorithms
 *
 * Created by domesc on 15/11/16.
 */
object Distance {

  val euclidean = (x: Vector[Double], y: Vector[Double]) => norm(x - y, 2)

  val manhattan = (x: Vector[Double], y: Vector[Double]) => sum(abs(x - y))

  val chebyshev = (x: Vector[Double], y: Vector[Double]) => max(abs(x - y))

  val minkowsky = (x: Vector[Double], y: Vector[Double], p: Double) => norm(x - y, p)

  val wminkowski = (x: Vector[Double], y: Vector[Double], p: Double, w: Double) => pow(w, 1 / p) * minkowsky.apply(x, y, p)

}
