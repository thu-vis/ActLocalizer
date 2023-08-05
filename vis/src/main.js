import Vue from 'vue'
import App from './App.vue'
import vuetify from './plugins/vuetify';
import store from "./store"
import Axios from 'axios'

Vue.config.productionTip = false

Vue.prototype.$axios = Axios

Axios.defaults.baseURL = '/api'

new Vue({
  vuetify,
  store,
  render: h => h(App)
}).$mount('#app')
