using Hachiko.Pages;
using Hachiko;
using System.Security.AccessControl;

namespace Hachiko;

public partial class MainPage : ContentPage {

    public AppPage CurrentPage { get; private set; }

    public MainPage() {
        InitializeComponent();
        ShowPage(new Home());
    }

    // DISPLAY VIEW  
    public void ShowPage(View newPage) {
        PageHost.Content = newPage;

        if (newPage is Home)
            NavBarRef.SetActive(AppPage.Home);
        else if (newPage is Object_)
            NavBarRef.SetActive(AppPage.Object);
        else if (newPage is Smart)
            NavBarRef.SetActive(AppPage.Smart);
    }



    // SWIPE NAVIGATION MECHANICS
    private async void OnSwipeLeft(object sender, SwipedEventArgs e) {
        var current = PageHost.Content;

        if (current is Home)
            await SlideTo(new Object_());

        else if (current is Object_)
            await SlideTo(new Smart());
    }

    private async void OnSwipeRight(object sender, SwipedEventArgs e) {
        var current = PageHost.Content;

        if (current is Smart)
            await SlideTo(new Object_());

        else if (current is Object_)
            await SlideTo(new Home());
    }



    // PAGES TRANSITION
    public async Task SlideTo(View newPage) {
        var oldView = PageHost.Content;
        
        int GetOrder(View v) {
            if (v is Home) return 0;
            if (v is Object_) return 1;
            if (v is Smart) return 2;
            return -1;
        }

        int oldSlideDir = 1;
        int newSlideDir = -1;

        if (oldView != null) {
            int oldOrder = GetOrder(oldView);
            int newOrder = GetOrder(newPage);

            if (newOrder > oldOrder) {
                oldSlideDir = -1;
                newSlideDir = 1;
            } else {
                oldSlideDir = 1;
                newSlideDir = -1;
            }

            // slide + fade out old page simultaneously
            await Task.WhenAll(
                oldView.TranslateTo(oldSlideDir * 300, 0, 200),
                oldView.FadeTo(0, 200)
            );
        }

        // prepare new page: off-screen + invisible
        newPage.TranslationX = newSlideDir * 300;
        newPage.Opacity = 0;
        PageHost.Content = newPage;

        // slide + fade in new page simultaneously
        await Task.WhenAll(
            newPage.TranslateTo(0, 0, 200),
            newPage.FadeTo(1, 200)
        );

        // reset old page state in case it's reused
        if (oldView != null) {
            oldView.Opacity = 1;
            oldView.TranslationX = 0;
        }

        ShowPage(newPage);
    }


}