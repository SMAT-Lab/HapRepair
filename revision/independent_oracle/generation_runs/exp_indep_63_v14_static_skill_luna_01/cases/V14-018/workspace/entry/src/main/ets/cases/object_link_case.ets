@Observed
class ProfileData {
  name: string;
  constructor(name: string) { this.name = name; }
}

@Component
struct ProfileName {
  @ObjectLink profile: ProfileData;
  build() { Text(this.profile.name) }
}

@Entry
@Component
struct ProfilePage {
  @State profiles: ProfileData[] = [new ProfileData('Lin')];
  build() { ProfileName({ profile: this.profiles[0] }) }
}
