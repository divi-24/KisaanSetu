import './App.css';
import { Routes, Route, useLocation } from 'react-router-dom';
import Navbar from './components/Navbar';
import Crop from './components/models/CropRecommendation';
import Home from './pages/Home';
import Contact from './pages/Contact';
import About from './pages/About';
import Disease from './components/Disease';
import Fertilizer from './components/models/Fertilizer';
import SoilQuality from './components/models/SoilQuality';
import Footer from './components/Footer';
import GoTop from './components/GoTop';
import NotFound from './NotFound';
import Prices from './components/models/Prices';
import Reports from './components/models/Reports';
import AboutUs from "./components/AboutUs";
import Contributor from './pages/ContributorsPage';
import UseScrollToTop from './components/UseScrollToTop';
import Article from './pages/Article';
import TaskReminder from './components/tools/TaskReminder';
import ChatBot from './pages/ChatBot';
import CropRotationRecommendation from './components/models/CropRotationRecommendation';
import DiseaseRecognition from './pages/Disease/DiseaseRecognition';
import SugarcaneRecognition from './pages/Disease/SugarcaneRecognition';
import PaddyRecognition from './pages/Disease/PaddyRecognition';
import Preloader from "./components/PreLoader";
import ProgressScrollDown from "./components/ProgressScrollDown";
import React, { useState, useEffect } from "react";
import Climate from './components/help/Climate';
import Products from "./pages/Products";
import AuthPage from './components/AuthPage';
import WhyAI from './pages/WhyAI'; // Import the WhyAI component
import LoginPage from './components/LoginPage';
import SignUpPage from './components/SignUpPage';
import { AuthProvider } from './context/AuthContext';
import TermsAndConditions from './components/TermsAndConditions';
import CookiePolicy from './components/CookiePolicy';
import PlantTaskReminder from './components/tools/PlantTaskReminder';
import CodeOfConduct from './components/CodeOfConduct';

import MushroomEdibility from './components/models/Mushroom';
import PrivacyPolicy from './components/PrivacyPolicy';
import Licensing from './components/Licensing';
import Feedback from './components/Feedback';
import SoilTestingCentres from './components/SoilTestingCenters';
import NewsForum from './components/NewsForum';
import ElectricalElectronicsShops from './components/ElectricalElectronicsShops';
import DiscussionPage from './components/Discussions';
//AgroRentAI
import HeroSectionRent from './AgroRentAI/HeroSectionRent';
import NavigateProducts from './AgroRentAI/NavigateProducts';
import RentUserDashboard from './AgroRentAI/RentUserDashboard';
import RentCheckoutPage from './AgroRentAI/RentCheckoutPage';
import RentCartPage from './AgroRentAI/Cart';

import RentProductDetails from './AgroRentAI/RentProductDetails';


import RentAdminDashboard from './AgroRentAI/RentAdminDashboard';

//AgroShopAI
import HomeShop from './AgroShopAI/pages/HomeShop';
import ShopFooter from './AgroShopAI/components/ShopFooter';
import CategoryPage from './AgroShopAI/pages/CategoryPage';
import ProductPage from './AgroShopAI/pages/ProductPage';
import BestPractices from './pages/BestPractices';
import Profile from './components/Profile';
import AgriProductListing from './AgroRentAI/components/AgriProductListing';
import CartPage from './AgroShopAI/pages/Cart';
import Wishlist from './AgroShopAI/pages/Wishlist';
import ShopNavbar from './AgroShopAI/components/ShopNavbar';
import ShopProfile from './AgroShopAI/pages/Profile';
import SearchResult from './AgroShopAI/pages/SearchResult'
import CancelAndReturnPolicy from './AgroShopAI/pages/FooterPages/CancelAndReturn';
import TermsOfUse from './AgroShopAI/pages/FooterPages/TermsOfUse';
import ShopPrivacyPolicy from './AgroShopAI/pages/FooterPages/Privacy';
import GrievanceRedressal from './AgroShopAI/pages/FooterPages/Grievance';
import ForgotPasswordPage from './components/ForgotPassword';
import AccountVerificationPage from './components/EmailVerification';

import FAQ from './AgroShopAI/pages/Faq';
import GeminiChat from './components/tools/GeminiChat';
import ResendVerificationPage from './components/ResendVerification';

import DiscussionForum from './components/DiscussionForum';

import AiChatbot from './components/AiChatbot';


import WaterManagement from './components/models/WaterManagement';

import RentSupportPage from './AgroRentAI/components/RentSupportPage';

const MainContent = () => {
  UseScrollToTop();
  const location = useLocation(); // Get the current route
  const [isPreloaderVisible, setIsPreloaderVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsPreloaderVisible(false);
    }, 5000); // Preloader visible for 5 seconds
    return () => clearTimeout(timer);
  }, []);

  // Check if the current path is the one you want to hide the Navbar for
  const normalizePath = (path) => path.toLowerCase().replace(/^\/+|\/+$/g, '');
  const hideNavbarRoutes = ['navigateproducts', '404'];
  const agroShopRoute = 'agroshop';
  const normalizedPath = normalizePath(location.pathname);
  const hideNavbar = hideNavbarRoutes.includes(normalizedPath) || normalizedPath.startsWith(agroShopRoute);
  const checkShop = normalizedPath.startsWith(agroShopRoute);
  return (
    <>
      {isPreloaderVisible ? (
        <Preloader />
      ) : (
        <div>
          <AuthProvider>
            <GoTop />
            <AiChatbot />
            <ProgressScrollDown />
            <div>
              {!hideNavbar ? <Navbar /> : <ShopNavbar />} {/* Conditional rendering for Navbar */}
              <Routes>
                <Route path="/thank-you" element={<Feedback />} /> {/* Thank You Page Route */}
                <Route path="/privacy-policy" element={<PrivacyPolicy />} />
                <Route path="/licensing" element={<Licensing />} />
                <Route path="/" element={<Home />} />
                <Route path="/chatbot" element={<ChatBot />} />
                <Route path="/contact" element={<Contact />} />
                <Route path="/contributor" element={<Contributor />} />
                <Route path="/about" element={<About />} />
                <Route path="/crop" element={<Crop />} />
                <Route path="/water-management" element={<WaterManagement />} />
                <Route path="/fertilizer" element={<Fertilizer />} />
                <Route path="/soil" element={<SoilQuality />} />
                <Route path="/disease" element={<Disease />} />
                <Route path="/crop_recommendation" element={<CropRotationRecommendation />} />
                <Route path="/code-of-conduct" element={<CodeOfConduct />} />

                <Route path="/prices" element={<Prices />} />
                <Route path="/reports" element={<Reports />} />
                <Route path="/aboutus" element={<AboutUs />} />
                <Route path="/article" element={<Article />} />
                <Route path="/soiltestingcentres" element={<SoilTestingCentres />} />

                <Route path="/TaskReminder" element={<TaskReminder />} />
                <Route path="/GeminiChat" element={<GeminiChat />} />
                <Route path="/SugarcaneRecognition" element={<SugarcaneRecognition />} />
                <Route path="/PaddyRecognition" element={<PaddyRecognition />} />
                <Route path="/DiseaseRecognition" element={<DiseaseRecognition />} />
                <Route path="/PlantTaskReminder" element={<PlantTaskReminder />} />


                <Route path="/Climate" element={<Climate />} />

                <Route path="/MushroomEdibility" element={<MushroomEdibility />} />
                <Route path="/products" element={<Products />} />
                <Route path="/Auth-page" element={<AuthPage />} />
                <Route path="/whyai" element={<WhyAI />} />
                <Route path="/login" element={<LoginPage />} />
                <Route path="/profile" element={<Profile />} />
                <Route path="/signup" element={<SignUpPage />} />
                <Route path="/verify-email" element={<AccountVerificationPage />} />
                <Route path="/verification" element={<ResendVerificationPage />} />
                <Route path="/forgot-password" element={<ForgotPasswordPage />} />
                <Route path="/terms" element={<TermsAndConditions />} />
                <Route path="/cookie-policy" element={<CookiePolicy />} />
                <Route path="/news" element={<NewsForum />} />
                <Route path="/ee-shops" element={<ElectricalElectronicsShops />} />
                <Route path="/BestPractices" element={<BestPractices />} />
                <Route path="/DiscussionPage" element={<DiscussionPage />} />
                {/* AgroRentAI Routes */}
                <Route path="/HeroSectionRent" element={<HeroSectionRent />} />
                <Route path="/NavigateProducts" element={<NavigateProducts />} />
                <Route path="/AgriProducts" element={<AgriProductListing />} />
                <Route path="/RentCheckoutPage" element={<RentCheckoutPage />} />
                <Route path="/RentCart" element={<RentCartPage />} />
                
                <Route path="/RentProductDetails/:productId" element={<RentProductDetails />} />

                <Route path="/RentAdminDashboard" element={<RentAdminDashboard />} />
                <Route path="/RentUserDashboard" element={<RentUserDashboard />} />
                <Route path="/RentSupport" element={<RentSupportPage />} />

                <Route path="*" element={<NotFound />} />
                {/* AgroShopAI Routes */}
                <Route path="/AgroShop" element={<HomeShop />} />
                <Route path="/AgroShop/Category" element={<CategoryPage />} />
                <Route path="/AgroShop/Category/:name" element={<CategoryPage />} />
                <Route path="/AgroShop/Product/:id" element={<ProductPage />} />
                <Route path="/AgroShop/Cart" element={<CartPage />} />
                <Route path="/AgroShop/Wishlist" element={<Wishlist />} />
                <Route path="/AgroShop/Profile" element={<ShopProfile />} />
                <Route path="/AgroShop/search" element={<SearchResult />} />
                {/* Footer Links */}
                <Route path="/AgroShop/cancellation-return" element={<CancelAndReturnPolicy />} />~
                <Route path="/AgroShop/terms-of-use" element={<TermsOfUse />} />
                <Route path="/AgroShop/privacy-policy" element={<ShopPrivacyPolicy />} />
                <Route path="/AgroShop/faq" element={<FAQ />} />
                <Route path="/AgroShop/grievance" element={<GrievanceRedressal />} />
                <Route path="/discussion" element={<DiscussionForum />} />


  
              </Routes>
              {checkShop ? <ShopFooter /> : <Footer />}
            </div>
          </AuthProvider>
        </div>
      )}
    </>
  );
};

export default MainContent;
