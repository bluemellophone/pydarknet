#ifndef DETECTOR_DLL_DEFINES_H
	#define DETECTOR_DLL_DEFINES_H

	#ifdef WIN32
	    #ifndef snprintf
	    	#define snprintf _snprintf
	    #endif
	#endif

	#define DARKNET_DETECTOR_EXPORT

	#ifndef FOO_DLL
	    #ifdef DARKNET_DETECTOR_EXPORTS
	        #define DARKNET_DETECTOR_EXPORT __declspec(dllexport)
	    #else
	        //#define DETECTOR_EXPORT __declspec(dllimport)
	    #endif
	#else
		#define DARKNET_DETECTOR_EXPORT
	#endif
#endif
