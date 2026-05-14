use std::f64::consts::PI;

use crate::model::Location;

const EARTH_RADIUS_KM: f64 = 6371.0;

#[inline(always)]
fn to_radians(degrees: f64) -> f64 {
  degrees * (PI / 180.0)
}

// Great-circle distance between two locations using the spherical law of cosines.
#[inline(always)]
pub fn haversine_km(from: &Location, to: &Location) -> f64 {
  let lat1 = to_radians(from.latitude);
  let lon1 = to_radians(from.longitude);
  let lat2 = to_radians(to.latitude);
  let lon2 = to_radians(to.longitude);

  let val = lat1.sin() * lat2.sin() + lat1.cos() * lat2.cos() * (lon1 - lon2).cos();
  let clamped = val.clamp(-1.0, 1.0);
  clamped.acos() * EARTH_RADIUS_KM
}

#[cfg(test)]
mod tests {
  use super::*;

  fn loc(hash: &str, latitude: f64, longitude: f64) -> Location {
    Location {
      hash: hash.to_string(),
      latitude,
      longitude,
    }
  }

  #[test]
  fn identical_points_have_zero_distance() {
    let a = loc("a", 54.6872, 25.2797);
    assert_eq!(haversine_km(&a, &a), 0.0);
  }

  #[test]
  fn vilnius_to_kaunas_is_about_91km() {
    let vilnius = loc("vilnius", 54.6872, 25.2797);
    let kaunas = loc("kaunas", 54.8985, 23.9036);
    let d = haversine_km(&vilnius, &kaunas);
    assert!((d - 91.3).abs() < 0.5, "expected ~91.3 km, got {d}");
  }

  #[test]
  fn distance_is_symmetric() {
    let a = loc("a", 10.0, 20.0);
    let b = loc("b", 30.0, 40.0);
    let ab = haversine_km(&a, &b);
    let ba = haversine_km(&b, &a);
    assert!((ab - ba).abs() < 1e-9);
  }
}
