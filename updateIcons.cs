public class Vehicle {
  public int doors;
  public int wheels;

  public Vehicle(int _doors, int _wheels) {
    doors = _doors;
    wheels = _wheels;
  }

}

public class Car : Vehicle
{
  Vehicle vehicle = new Vehicle(4, 4);
}