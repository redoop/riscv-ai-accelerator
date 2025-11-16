ThisBuild / scalaVersion := "2.13.12"
ThisBuild / version      := "0.1.0"
ThisBuild / organization := "org.chipsalliance"

val chiselVersion = "6.0.0"

lazy val root = (project in file("."))
  .settings(
    name := "chisel-matrix-multiplier",
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel"     % chiselVersion,
      "edu.berkeley.cs"   %% "chiseltest" % chiselVersion % Test,
      "org.scalatest"     %% "scalatest"  % "3.2.16"      % Test
    ),
    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-deprecation",
      "-feature",
      "-Xcheckinit",
      "-Xfatal-warnings",
      "-Ywarn-dead-code",
      "-Ywarn-unused",
      "-Ymacro-annotations"
    ),
    addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full),
    
    // Test configuration
    Test / testOptions += Tests.Argument(TestFrameworks.ScalaTest, "-oD"),
    Test / parallelExecution := false,
    Test / logBuffered := false
  )