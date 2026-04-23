using System.Security.AccessControl;
using Hachiko.Pages;
namespace Hachiko.Component;

public partial class NavBar : ContentView {
	public NavBar()
	{
		InitializeComponent();

	}

	private async void Home_Clicked(object sender, EventArgs e) {
		await ((MainPage)Application.Current.MainPage)
			.SlideTo(new Home());
	}

	private async void Object_Clicked(object sender, EventArgs e) {
		await ((MainPage)Application.Current.MainPage)
			.SlideTo(new Object_());
	}

	private async void Smart_Clicked(object sender, EventArgs e) {
		await ((MainPage)Application.Current.MainPage)
			.SlideTo(new Smart());
	}


	public void SetActive(AppPage page) {
		HomeIcon.Source = page == AppPage.Home
			? "home_ico.png"
			: "home_in_ico.png";

		ObjectIcon.Source = page == AppPage.Object
			? "obj_ico.png"
			: "obj_in_ico.png";

		SmartIcon.Source = page == AppPage.Smart
			? "smart_ico.png"
			: "smart_in_ico.png";

		HomeFrame.BackgroundColor = page == AppPage.Home
			? Color.FromArgb("#e4e8ed")
			: Colors.Transparent;

		ObjectFrame.BackgroundColor = page == AppPage.Object
			? Color.FromArgb("#e4e8ed")
			: Colors.Transparent;

		SmartFrame.BackgroundColor = page == AppPage.Smart
			? Color.FromArgb("#e4e8ed")
			: Colors.Transparent;

	}
}