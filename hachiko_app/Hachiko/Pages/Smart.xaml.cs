namespace Hachiko.Pages;

public partial class Smart : ContentView {
	public Smart()
	{
		InitializeComponent();
        ShowPage(new SmartHome());

    }

    public void ShowPage(View newPage) {
        SmartHost.Content = newPage;
    }
}